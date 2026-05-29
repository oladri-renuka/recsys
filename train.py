import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from typing import Dict
from tqdm import tqdm

from data import preprocess
from dataset import create_dataloaders
from model import SASRec


class SASRecLoss(nn.Module):
    """SASRec loss: Binary cross-entropy."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, target_ids, neg_ids, mask):
        pos_logits = logits.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        neg_logits = logits.gather(2, neg_ids.unsqueeze(-1)).squeeze(-1)
        
        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits), reduction='none'
        )
        
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits), reduction='none'
        )
        
        loss_per_pos = (pos_loss + neg_loss) * mask
        num_valid = mask.sum()
        if num_valid == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        return loss_per_pos.sum() / num_valid


def train_epoch(model, train_loader, loss_fn, optimizer, device):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(train_loader, desc="  Training", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        neg_ids = batch['neg_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        
        logits = model(input_ids, mask)
        loss = loss_fn(logits, target_ids, neg_ids, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return epoch_loss / len(train_loader)


def evaluate(model, val_loader, train_set, num_items, device, k=10):
    """Evaluate model."""
    model.eval()
    
    user_history = {}
    for record in train_set:
        user_id = record['user_id']
        if user_id not in user_history:
            user_history[user_id] = set()
        user_history[user_id].update(record['input_seq'])
    
    all_hits = []
    all_ndcgs = []
    
    pbar = tqdm(val_loader, desc="  Evaluating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            user_ids = batch['user_ids'].to(device)
            
            logits = model(input_ids, mask)
            logits_np = logits.cpu().numpy()
            target_ids_np = target_ids.cpu().numpy()
            user_ids_np = user_ids.cpu().numpy()
            
            for i in range(len(target_ids_np)):
                user_id = user_ids_np[i]
                user_logits = logits_np[i]
                target_item = target_ids_np[i]
                
                rng = np.random.RandomState(user_id)
                seen_items = user_history.get(user_id, set())
                candidates = list(set(range(1, num_items + 1)) - seen_items)
                
                if len(candidates) >= 100:
                    neg_items = rng.choice(candidates, size=100, replace=False)
                else:
                    neg_items = rng.choice(candidates, size=100, replace=True)
                
                ranking_items = [target_item] + list(neg_items)
                pred_logits = user_logits[-1]
                ranking_logits = pred_logits[ranking_items]
                
                sorted_idx = np.argsort(-ranking_logits)
                target_rank = np.where(sorted_idx == 0)[0][0]
                
                hit = 1 if target_rank < k else 0
                all_hits.append(hit)
                
                rank = target_rank + 1
                dcg = 1.0 / np.log2(rank + 1)
                ndcg = dcg / np.log2(2)
                all_ndcgs.append(ndcg)
            
            pbar.update(1)
    
    return {
        'hit_at_k': np.mean(all_hits) if all_hits else 0.0,
        'ndcg_at_k': np.mean(all_ndcgs) if all_ndcgs else 0.0
    }


def train(num_epochs=500, batch_size=128, learning_rate=0.001):
    """Main training loop."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path("results")
    checkpoint_dir.mkdir(exist_ok=True)
    log_file = checkpoint_dir / "training_log.json"
    
    print(f"\nTraining on device: {device}")
    print(f"Saving to: {checkpoint_dir}\n")
    
    # Load data
    print("Loading preprocessed data...")
    train_set, val_set, test_set, stats = preprocess("ml-1m")
    num_items = stats['num_items']
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_set, val_set, test_set,
        num_items=num_items,
        batch_size=batch_size,
        max_len=200
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}\n")
    
    # Create model
    print("Creating SASRec model...")
    model = SASRec(
        num_items=num_items,
        d_model=50,
        num_blocks=2,
        num_heads=1,
        dropout=0.2,
        max_len=200
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")
    
    # Loss and optimizer
    loss_fn = SASRecLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    print(f"Starting training (max epochs={num_epochs}, patience=100)...")
    print("=" * 80)
    
    training_log = []
    best_val_ndcg = 0.0
    best_epoch = 0
    patience = 100
    no_improve = 0
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader, train_set, num_items, device, k=10)
        val_hit = val_metrics['hit_at_k']
        val_ndcg = val_metrics['ndcg_at_k']
        
        training_log.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_hit_at_10': float(val_hit),
            'val_ndcg_at_10': float(val_ndcg)
        })
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Hit@10: {val_hit:.4f} | NDCG@10: {val_ndcg:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            if epoch % 10 == 0 or epoch == 1:
                print(f"           ✓ new best NDCG: {val_ndcg:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}: no improvement for {patience} epochs")
                print(f"Best validation NDCG: {best_val_ndcg:.4f} at epoch {best_epoch}")
                break
        
        scheduler.step()
    
    print("=" * 80)
    
    # Test evaluation
    print(f"\nEvaluating on test set...")
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt"))
    test_metrics = evaluate(model, test_loader, train_set, num_items, device, k=10)
    test_hit = test_metrics['hit_at_k']
    test_ndcg = test_metrics['ndcg_at_k']
    
    print(f"Test Hit@10:  {test_hit:.4f}")
    print(f"Test NDCG@10: {test_ndcg:.4f}")
    
    # Save logs
    with open(log_file, 'w') as f:
        json.dump({
            'config': {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'd_model': 50,
                'num_blocks': 2,
                'dropout': 0.2,
                'num_heads': 1,
                'max_seq_len': 200
            },
            'final_results': {
                'test_hit_at_10': float(test_hit),
                'test_ndcg_at_10': float(test_ndcg),
                'best_val_ndcg': float(best_val_ndcg),
                'best_epoch': int(best_epoch)
            },
            'training_log': training_log
        }, f, indent=2)
    
    print(f"\nLogs saved to {log_file}")
    
    return {
        'test_hit_at_10': test_hit,
        'test_ndcg_at_10': test_ndcg,
        'best_val_ndcg': best_val_ndcg,
        'best_epoch': best_epoch
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SASREC TRAINING - LOCAL")
    print("="*80 + "\n")
    
    results = train(num_epochs=500, batch_size=128, learning_rate=0.001)
    
    print(f"\n{'=' * 80}")
    print(f"FINAL RESULTS")
    print(f"Paper targets: Hit@10=0.8245, NDCG@10=0.5905")
    print(f"{'=' * 80}")
    print(f"Test Hit@10:  {results['test_hit_at_10']:.4f}")
    print(f"Test NDCG@10: {results['test_ndcg_at_10']:.4f}")
    print(f"Best epoch:   {results['best_epoch']}")
    print(f"{'=' * 80}")