import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from tqdm import tqdm

from data import preprocess
from dataset import create_dataloaders
from model import SASRec, SASRecBlock, MultiHeadAttention


class SASRecLoss(nn.Module):
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


class SASRecNoPositionalEmbedding(nn.Module):
    def __init__(self, num_items, d_model=50, num_blocks=2, num_heads=1, dropout=0.2, max_len=200):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len
        
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.blocks = nn.ModuleList([SASRecBlock(d_model, num_heads, dropout) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, input_ids, attention_mask):
        x = self.item_embedding(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.final_norm(x)
        logits = torch.matmul(x, self.item_embedding.weight.T)
        return logits


class SASRecOneBlock(nn.Module):
    def __init__(self, num_items, d_model=50, num_heads=1, dropout=0.2, max_len=200):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len
        
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.positional_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([SASRecBlock(d_model, num_heads, dropout) for _ in range(1)])
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, input_ids, attention_mask):
        x = self.item_embedding(input_ids)
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = x + self.positional_embedding(positions)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.final_norm(x)
        logits = torch.matmul(x, self.item_embedding.weight.T)
        return logits


def train_epoch(model, train_loader, loss_fn, optimizer, device):
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


def evaluate(model, val_loader, train_set, num_items, device, segment_users=None, k=10):
    """
    Evaluate model, optionally filtered to specific user segment.
    
    Args:
        segment_users: Set of user IDs to evaluate (None = all users)
    """
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
                
                # Filter to segment if specified
                if segment_users is not None and user_id not in segment_users:
                    continue
                
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


def segment_users(train_set):
    """Segment users by sequence length."""
    segments = {
        'short': set(),
        'medium': set(),
        'long': set()
    }
    
    for record in train_set:
        user_id = record['user_id']
        seq_len = len(record['input_seq'])
        if seq_len < 20:
            segments['short'].add(user_id)
        elif seq_len <= 100:
            segments['medium'].add(user_id)
        else:
            segments['long'].add(user_id)
    
    return segments


def run_ablation(num_epochs=100, batch_size=128, learning_rate=0.001):
    """Run ablation studies including per-segment PE analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path("results")
    
    print("Loading preprocessed data...")
    train_set, val_set, test_set, stats = preprocess("ml-1m")
    num_items = stats['num_items']
    
    print("Segmenting users...")
    segments = segment_users(train_set)
    print(f"SHORT: {len(segments['short'])} users")
    print(f"MEDIUM: {len(segments['medium'])} users")
    print(f"LONG: {len(segments['long'])} users")
    
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_set, val_set, test_set,
        num_items=num_items,
        batch_size=batch_size,
        max_len=200
    )
    
    ablation_results = {}
    
    # === CONDITION 1: DEFAULT (WITH PE) ===
    print("\n" + "="*80)
    print("ABLATION 1: DEFAULT MODEL (2 blocks, WITH Positional Embedding)")
    print("="*80)
    model_default = SASRec(num_items, d_model=50, num_blocks=2, num_heads=1, dropout=0.2, max_len=200).to(device)
    model_default.load_state_dict(torch.load(checkpoint_dir / "best_model.pt"))
    
    print("Evaluating on test set...")
    test_metrics_overall = evaluate(model_default, test_loader, train_set, num_items, device, k=10)
    
    ablation_results['default'] = {
        'hit_at_10': float(test_metrics_overall['hit_at_k']),
        'ndcg_at_10': float(test_metrics_overall['ndcg_at_k']),
        'description': '2 blocks, positional embedding',
        'per_segment': {}
    }
    
    # Evaluate per segment
    for seg_name, seg_users in segments.items():
        metrics = evaluate(model_default, test_loader, train_set, num_items, device, segment_users=seg_users, k=10)
        ablation_results['default']['per_segment'][seg_name] = {
            'hit_at_10': float(metrics['hit_at_k']),
            'ndcg_at_10': float(metrics['ndcg_at_k']),
            'num_users': len(seg_users)
        }
    
    print(f"Overall - Hit@10: {test_metrics_overall['hit_at_k']:.4f}, NDCG@10: {test_metrics_overall['ndcg_at_k']:.4f}")
    for seg_name, seg_data in ablation_results['default']['per_segment'].items():
        print(f"  {seg_name.upper()}: Hit@10={seg_data['hit_at_10']:.4f}, NDCG@10={seg_data['ndcg_at_10']:.4f}")
    
    # === CONDITION 2: NO PE ===
    print("\n" + "="*80)
    print("ABLATION 2: NO POSITIONAL EMBEDDING (2 blocks)")
    print("="*80)
    model_no_pe = SASRecNoPositionalEmbedding(num_items, d_model=50, num_blocks=2, num_heads=1, dropout=0.2, max_len=200).to(device)
    loss_fn = SASRecLoss()
    optimizer = optim.Adam(model_no_pe.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    best_val_ndcg = 0.0
    best_epoch = 0
    patience = 15
    no_improve = 0
    
    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training No-PE Model", unit="epoch")
    for epoch in epoch_bar:
        train_loss = train_epoch(model_no_pe, train_loader, loss_fn, optimizer, device)
        
        if epoch % 10 == 0 or epoch == 1:
            val_metrics = evaluate(model_no_pe, val_loader, train_set, num_items, device, k=10)
            val_ndcg = val_metrics['ndcg_at_k']
            
            epoch_bar.set_postfix({
                "loss": f"{train_loss:.4f}",
                "val_ndcg": f"{val_ndcg:.4f}",
                "best": f"{best_val_ndcg:.4f}"
            })
            
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    epoch_bar.close()
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            epoch_bar.set_postfix({
                "loss": f"{train_loss:.4f}",
                "best": f"{best_val_ndcg:.4f}"
            })
        
        scheduler.step()
    
    print("Final evaluation on test set...")
    test_metrics_no_pe = evaluate(model_no_pe, test_loader, train_set, num_items, device, k=10)
    
    ablation_results['no_pe'] = {
        'hit_at_10': float(test_metrics_no_pe['hit_at_k']),
        'ndcg_at_10': float(test_metrics_no_pe['ndcg_at_k']),
        'description': '2 blocks, NO positional embedding',
        'best_epoch': best_epoch,
        'per_segment': {}
    }
    
    # Evaluate per segment
    for seg_name, seg_users in segments.items():
        metrics = evaluate(model_no_pe, test_loader, train_set, num_items, device, segment_users=seg_users, k=10)
        ablation_results['no_pe']['per_segment'][seg_name] = {
            'hit_at_10': float(metrics['hit_at_k']),
            'ndcg_at_10': float(metrics['ndcg_at_k']),
            'num_users': len(seg_users),
            'pe_impact_ndcg_pct': float(
                (ablation_results['default']['per_segment'][seg_name]['ndcg_at_10'] - metrics['ndcg_at_k']) / 
                ablation_results['default']['per_segment'][seg_name]['ndcg_at_10'] * 100
            )
        }
    
    print(f"Overall - Hit@10: {test_metrics_no_pe['hit_at_k']:.4f}, NDCG@10: {test_metrics_no_pe['ndcg_at_k']:.4f}")
    for seg_name, seg_data in ablation_results['no_pe']['per_segment'].items():
        print(f"  {seg_name.upper()}: Hit@10={seg_data['hit_at_10']:.4f}, NDCG@10={seg_data['ndcg_at_10']:.4f}, PE Impact={seg_data['pe_impact_ndcg_pct']:.1f}%")
    
    # === CONDITION 3: ONE BLOCK ===
    print("\n" + "="*80)
    print("ABLATION 3: ONE BLOCK (1 block, WITH PE)")
    print("="*80)
    model_one_block = SASRecOneBlock(num_items, d_model=50, num_heads=1, dropout=0.2, max_len=200).to(device)
    optimizer = optim.Adam(model_one_block.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    best_val_ndcg = 0.0
    best_epoch = 0
    no_improve = 0
    
    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training 1-Block Model", unit="epoch")
    for epoch in epoch_bar:
        train_loss = train_epoch(model_one_block, train_loader, loss_fn, optimizer, device)
        
        if epoch % 10 == 0 or epoch == 1:
            val_metrics = evaluate(model_one_block, val_loader, train_set, num_items, device, k=10)
            val_ndcg = val_metrics['ndcg_at_k']
            
            epoch_bar.set_postfix({
                "loss": f"{train_loss:.4f}",
                "val_ndcg": f"{val_ndcg:.4f}",
                "best": f"{best_val_ndcg:.4f}"
            })
            
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    epoch_bar.close()
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            epoch_bar.set_postfix({
                "loss": f"{train_loss:.4f}",
                "best": f"{best_val_ndcg:.4f}"
            })
        
        scheduler.step()
    
    print("Final evaluation on test set...")
    test_metrics_one_block = evaluate(model_one_block, test_loader, train_set, num_items, device, k=10)
    
    ablation_results['one_block'] = {
        'hit_at_10': float(test_metrics_one_block['hit_at_k']),
        'ndcg_at_10': float(test_metrics_one_block['ndcg_at_k']),
        'description': '1 block, positional embedding',
        'best_epoch': best_epoch,
        'per_segment': {}
    }
    
    # Evaluate per segment
    for seg_name, seg_users in segments.items():
        metrics = evaluate(model_one_block, test_loader, train_set, num_items, device, segment_users=seg_users, k=10)
        ablation_results['one_block']['per_segment'][seg_name] = {
            'hit_at_10': float(metrics['hit_at_k']),
            'ndcg_at_10': float(metrics['ndcg_at_k']),
            'num_users': len(seg_users)
        }
    
    print(f"Overall - Hit@10: {test_metrics_one_block['hit_at_k']:.4f}, NDCG@10: {test_metrics_one_block['ndcg_at_k']:.4f}")
    for seg_name, seg_data in ablation_results['one_block']['per_segment'].items():
        print(f"  {seg_name.upper()}: Hit@10={seg_data['hit_at_10']:.4f}, NDCG@10={seg_data['ndcg_at_10']:.4f}")
    
    # Save results
    with open(checkpoint_dir / "ablation_results.json", 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Hit@10':<12} {'NDCG@10':<12} {'vs Default':<15}")
    print("-" * 64)
    default_ndcg = ablation_results['default']['ndcg_at_10']
    for name, results in ablation_results.items():
        diff = ((results['ndcg_at_10'] - default_ndcg) / default_ndcg * 100) if name != 'default' else 0
        diff_str = f"{diff:+.2f}%" if name != 'default' else "baseline"
        print(f"{name:<25} {results['hit_at_10']:<12.4f} {results['ndcg_at_10']:<12.4f} {diff_str:<15}")
    
    print("\n" + "="*80)
    print("POSITIONAL EMBEDDING IMPACT PER SEGMENT")
    print("="*80)
    print("This analysis shows that positional encoding has different effects on different user segments")
    print(f"{'Segment':<15} {'Users':<10} {'With PE':<12} {'Without PE':<12} {'PE Impact':<15}")
    print("-" * 64)
    for seg_name in ['short', 'medium', 'long']:
        with_pe = ablation_results['default']['per_segment'][seg_name]['ndcg_at_10']
        without_pe = ablation_results['no_pe']['per_segment'][seg_name]['ndcg_at_10']
        impact = ablation_results['no_pe']['per_segment'][seg_name]['pe_impact_ndcg_pct']
        num_users = ablation_results['no_pe']['per_segment'][seg_name]['num_users']
        print(f"{seg_name:<15} {num_users:<10} {with_pe:<12.4f} {without_pe:<12.4f} {impact:+.1f}%")
    
    print("\nKey Finding: Positional encoding impact varies by user segment.")
    print("Short-history users may rely less on position information than long-history users.")
    
    print("\n" + "="*80)
    print("Ablation results saved to results/ablation_results.json")
    
    return ablation_results


if __name__ == "__main__":
    results = run_ablation(num_epochs=100, batch_size=128, learning_rate=0.001)