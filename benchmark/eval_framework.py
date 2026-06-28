"""
Full evaluation framework for SASRec.

Computes Precision@K, Recall@K, NDCG@K, Hit@K across multiple K values,
plus catalog coverage and popularity bias metrics.

Uses full-ranking (all items scored) rather than sampled negatives,
giving exact metrics instead of approximations.
"""
import sys
import os
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import SASRec
from data import preprocess
from dataset import create_dataloaders

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
K_VALUES = [1, 5, 10, 20, 50]


def get_item_popularity(train_set):
    """Count how often each item appears in training sequences."""
    counts = Counter()
    for record in train_set:
        counts.update(record["input_seq"])
    return counts


def evaluate_full_ranking(model, test_set, train_set, num_items, device, k_values=K_VALUES):
    """
    Full-ranking evaluation: score every item for each test user,
    mask out training items, rank, and compute metrics at each K.
    """
    model.eval()
    max_len = 200

    # Build per-user training history for masking
    user_history = {}
    for record in train_set:
        uid = record["user_id"]
        if uid not in user_history:
            user_history[uid] = set()
        user_history[uid].update(record["input_seq"])

    results = {k: {"hits": [], "precisions": [], "recalls": [], "ndcgs": []} for k in k_values}
    recommendation_counts = Counter()
    all_ranks = []

    t0 = time.time()

    with torch.no_grad():
        for i, record in enumerate(test_set):
            user_id = record["user_id"]
            input_seq = record["input_seq"]
            target_item = record["target_item"]

            # Pad and create tensors
            if len(input_seq) > max_len:
                input_seq = input_seq[-max_len:]
            pad_len = max_len - len(input_seq)
            padded = [0] * pad_len + input_seq
            mask = [0.0] * pad_len + [1.0] * len(input_seq)

            input_ids = torch.tensor([padded], dtype=torch.long).to(device)
            attn_mask = torch.tensor([mask], dtype=torch.float).to(device)

            logits = model(input_ids, attn_mask)
            scores = logits[0, -1, :].cpu().numpy()  # [num_items + 1]

            # Mask out padding (index 0) and training history
            scores[0] = -np.inf
            seen = user_history.get(user_id, set())
            for item_id in seen:
                if item_id != target_item:
                    scores[item_id] = -np.inf

            # Full ranking
            ranked_items = np.argsort(-scores)
            target_rank = int(np.where(ranked_items == target_item)[0][0])
            all_ranks.append(target_rank + 1)

            # Track recommendations for coverage/bias
            recommendation_counts.update(ranked_items[:10].tolist())

            for k in k_values:
                top_k = set(ranked_items[:k].tolist())
                hit = 1 if target_item in top_k else 0

                results[k]["hits"].append(hit)
                results[k]["precisions"].append(hit / k)
                results[k]["recalls"].append(hit)  # 1 relevant item

                if hit:
                    ndcg = 1.0 / np.log2(target_rank + 2)
                else:
                    ndcg = 0.0
                results[k]["ndcgs"].append(ndcg)

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  Evaluated {i+1}/{len(test_set)} users ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    n = len(test_set)
    print(f"  Full ranking evaluation: {n} users in {elapsed:.1f}s")

    metrics = {}
    for k in k_values:
        metrics[f"Hit@{k}"] = float(np.mean(results[k]["hits"]))
        metrics[f"Precision@{k}"] = float(np.mean(results[k]["precisions"]))
        metrics[f"Recall@{k}"] = float(np.mean(results[k]["recalls"]))
        metrics[f"NDCG@{k}"] = float(np.mean(results[k]["ndcgs"]))

    # Coverage: fraction of catalog in top-10 recommendations
    metrics["Coverage@10"] = len(recommendation_counts) / num_items

    # Popularity bias
    item_popularity = get_item_popularity(train_set)
    top_20pct_threshold = len(item_popularity) // 5
    top_20pct_items = set(
        sorted(item_popularity, key=item_popularity.get, reverse=True)[:top_20pct_threshold]
    )
    popular_recs = sum(
        count for item, count in recommendation_counts.items() if item in top_20pct_items
    )
    total_recs = sum(recommendation_counts.values())
    metrics["PopularityBias@10"] = popular_recs / total_recs if total_recs > 0 else 0.0

    # Mean Reciprocal Rank
    metrics["MRR"] = float(np.mean([1.0 / r for r in all_ranks]))

    # Median rank
    metrics["MedianRank"] = float(np.median(all_ranks))

    return metrics, recommendation_counts, item_popularity


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading data...")
    train_set, val_set, test_set, stats = preprocess("ml-1m")
    num_items = stats["num_items"]
    print(f"  {len(test_set)} test users, {num_items} items")

    print("\nLoading model...")
    model = SASRec(
        num_items=num_items, d_model=50, num_blocks=2,
        num_heads=1, dropout=0.2, max_len=200,
    ).to(device)
    model_path = RESULTS_DIR / "best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"  Loaded from {model_path}")

    print("\nRunning full-ranking evaluation...")
    metrics, rec_counts, item_pop = evaluate_full_ranking(
        model, test_set, train_set, num_items, device
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Full Ranking)")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'K=1':>8} {'K=5':>8} {'K=10':>8} {'K=20':>8} {'K=50':>8}")
    print("-" * 60)
    for metric_type in ["Hit", "Precision", "Recall", "NDCG"]:
        row = f"{metric_type:<20}"
        for k in K_VALUES:
            key = f"{metric_type}@{k}"
            row += f" {metrics[key]*100:>7.2f}%"
        print(row)

    print(f"\n{'MRR':<20} {metrics['MRR']*100:>7.2f}%")
    print(f"{'Median Rank':<20} {metrics['MedianRank']:>7.1f}")
    print(f"{'Coverage@10':<20} {metrics['Coverage@10']*100:>7.2f}%")
    print(f"{'PopularityBias@10':<20} {metrics['PopularityBias@10']*100:>7.2f}%")
    print("=" * 60)

    # Save results
    output = {
        "metrics": metrics,
        "k_values": K_VALUES,
        "num_test_users": len(test_set),
        "num_items": num_items,
        "recommendation_counts": {str(k): v for k, v in rec_counts.most_common(100)},
        "item_popularity_top100": {
            str(k): v for k, v in sorted(item_pop.items(), key=lambda x: -x[1])[:100]
        },
    }

    out_path = RESULTS_DIR / "eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    return output


if __name__ == "__main__":
    run_evaluation()
