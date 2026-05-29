import numpy as np
import torch
from pathlib import Path
import json
from tqdm import tqdm

from data import preprocess
from dataset import create_dataloaders
from model import SASRec


def segment_users(train_set):
    """Segment users by sequence length."""
    segments = {
        'short': [],      # < 20
        'medium': [],     # 20-100
        'long': []        # > 100
    }
    
    for record in train_set:
        seq_len = len(record['input_seq'])
        if seq_len < 20:
            segments['short'].append(record['user_id'])
        elif seq_len <= 100:
            segments['medium'].append(record['user_id'])
        else:
            segments['long'].append(record['user_id'])
    
    return segments


def evaluate_segment(model, val_loader, train_set, num_items, device, segment_users_set, k=10):
    """Evaluate metrics for a specific user segment."""
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
                
                # Only evaluate users in this segment
                if user_id not in segment_users_set:
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
        'ndcg_at_k': np.mean(all_ndcgs) if all_ndcgs else 0.0,
        'num_users': len(segment_users_set)
    }


def run_segment_analysis():
    """Analyze model performance by user segment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path("results")
    
    print("Loading preprocessed data...")
    train_set, val_set, test_set, stats = preprocess("ml-1m")
    num_items = stats['num_items']
    
    print("Segmenting users by sequence length...")
    segments = segment_users(train_set)
    
    for seg_name, user_ids in segments.items():
        print(f"{seg_name.upper()}: {len(user_ids)} users")
    
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_set, val_set, test_set,
        num_items=num_items,
        batch_size=128,
        max_len=200
    )
    
    print("Loading trained model...")
    model = SASRec(num_items, d_model=50, num_blocks=2, num_heads=1, dropout=0.2, max_len=200).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt"))
    
    # Evaluate per segment
    print("\n" + "="*80)
    print("SEGMENT ANALYSIS: TEST SET")
    print("="*80)
    
    segment_results = {}
    
    for seg_name, user_ids in segments.items():
        user_ids_set = set(user_ids)
        metrics = evaluate_segment(model, test_loader, train_set, num_items, device, user_ids_set, k=10)
        segment_results[seg_name] = {
            'num_users': metrics['num_users'],
            'hit_at_10': float(metrics['hit_at_k']),
            'ndcg_at_10': float(metrics['ndcg_at_k'])
        }
        print(f"\n{seg_name.upper()} ({metrics['num_users']} users):")
        print(f"  Hit@10:  {metrics['hit_at_k']:.4f}")
        print(f"  NDCG@10: {metrics['ndcg_at_k']:.4f}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Segment':<15} {'# Users':<12} {'Hit@10':<12} {'NDCG@10':<12}")
    print("-" * 51)
    for seg_name, results in segment_results.items():
        print(f"{seg_name:<15} {results['num_users']:<12} {results['hit_at_10']:<12.4f} {results['ndcg_at_10']:<12.4f}")
    
    # Save results
    with open(checkpoint_dir / "segment_analysis.json", 'w') as f:
        json.dump(segment_results, f, indent=2)
    
    print("\n✓ Segment analysis saved to results/segment_analysis.json")
    
    return segment_results


if __name__ == "__main__":
    results = run_segment_analysis()