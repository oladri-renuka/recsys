import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def preprocess(dataset_path="ml-1m"):
    """
    Preprocess MovieLens-1M dataset with k-core filtering.
    
    Args:
        dataset_path: Path to ml-1m folder containing ratings.dat
    
    Returns:
        train_set, val_set, test_set, stats dict
    """
    
    ratings_path = Path(dataset_path) / "ratings.dat"
    
    if not ratings_path.exists():
        raise FileNotFoundError(f"ratings.dat not found at {ratings_path}")
    
    print("\n" + "="*80)
    print("MOVIELEN-1M PREPROCESSING PIPELINE")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load raw ratings
    # ========================================================================
    print("\n[STEP 1] Raw ratings loaded:")
    
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        header=None,
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        engine='python'
    )
    
    print(f"  Interactions: {len(ratings)}")
    print(f"  Users: {ratings['user_id'].nunique()}")
    print(f"  Items: {ratings['item_id'].nunique()}")
    
    # ========================================================================
    # STEP 2: K-core filtering (k=5, user-first)
    # ========================================================================
    print("\n[STEP 2] K-core filtering (k=5)...")
    
    k = 5
    iteration = 0
    prev_users = len(ratings['user_id'].unique())
    prev_items = len(ratings['item_id'].unique())
    
    while True:
        iteration += 1
        
        # User-first: remove users with < k interactions
        user_counts = ratings.groupby('user_id').size()
        valid_users = user_counts[user_counts >= k].index
        ratings = ratings[ratings['user_id'].isin(valid_users)]
        
        # Item-first: remove items with < k interactions
        item_counts = ratings.groupby('item_id').size()
        valid_items = item_counts[item_counts >= k].index
        ratings = ratings[ratings['item_id'].isin(valid_items)]
        
        curr_users = len(ratings['user_id'].unique())
        curr_items = len(ratings['item_id'].unique())
        
        print(f"  Iteration {iteration}: {prev_users} → {curr_users} users | {prev_items} → {curr_items} items")
        
        if curr_users == prev_users and curr_items == prev_items:
            print(f"  K-core converged after {iteration} iterations")
            break
        
        prev_users = curr_users
        prev_items = curr_items
    
    print(f"  Post k-core: {curr_users} users | {curr_items} items | {len(ratings)} interactions")
    
    # ========================================================================
    # STEP 3: Remap IDs to [1, num_users] and [1, num_items]
    # ========================================================================
    print("\n[STEP 3] Remapping IDs...")
    
    user_map = {old_id: new_id for new_id, old_id in enumerate(sorted(ratings['user_id'].unique()), 1)}
    item_map = {old_id: new_id for new_id, old_id in enumerate(sorted(ratings['item_id'].unique()), 1)}
    
    ratings['user_id'] = ratings['user_id'].map(user_map)
    ratings['item_id'] = ratings['item_id'].map(item_map)
    
    num_users = len(user_map)
    num_items = len(item_map)
    
    print(f"  User IDs remapped: [1, {num_users}]")
    print(f"  Item IDs remapped: [1, {num_items}]")
    
    # ========================================================================
    # STEP 4: Build sequences (sorted by timestamp)
    # ========================================================================
    print("\n[STEP 4] Building sequences...")
    
    ratings = ratings.sort_values(['user_id', 'timestamp'])
    sequences = defaultdict(list)
    
    for _, row in ratings.iterrows():
        sequences[row['user_id']].append(row['item_id'])
    
    print(f"  {len(sequences)} users with ordered sequences")
    
    # ========================================================================
    # STEP 5: Filter sequences with <3 interactions
    # ========================================================================
    print("\n[STEP 5] Filtering sequences with <3 interactions...")
    
    num_before = len(sequences)
    sequences = {uid: seq for uid, seq in sequences.items() if len(seq) >= 3}
    num_after = len(sequences)
    
    print(f"  {num_before} → {num_after} users (discarded {num_before - num_after})")
    
    # ========================================================================
    # STEP 6: Train/Val/Test split (leave-one-out)
    # ========================================================================
    print("\n[STEP 6] Building train/val/test splits...")
    
    train_set = []
    val_set = []
    test_set = []
    
    for user_id, sequence in sequences.items():
        if len(sequence) < 3:
            continue
        
        # Split: test=last, val=second-to-last, train=rest
        train_items = sequence[:-2]
        val_item = sequence[-2]
        test_item = sequence[-1]
        
        train_set.append({
            'user_id': user_id,
            'input_seq': train_items,
            'target_seq': train_items
        })
        
        val_set.append({
            'user_id': user_id,
            'input_seq': train_items,
            'target_item': val_item
        })
        
        test_set.append({
            'user_id': user_id,
            'input_seq': train_items + [val_item],
            'target_item': test_item
        })
    
    print(f"  Train examples: {len(train_set)}")
    print(f"  Val examples: {len(val_set)}")
    print(f"  Test examples: {len(test_set)}")
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL DATASET STATISTICS")
    print("="*80)
    
    all_seq_lens = [len(s['input_seq']) for s in train_set]
    
    stats = {
        'num_users': num_users,
        'num_items': num_items,
        'num_interactions': len(ratings),
        'avg_seq_len': np.mean(all_seq_lens),
        'min_seq_len': np.min(all_seq_lens),
        'max_seq_len': np.max(all_seq_lens),
        'sparsity': 100 * (1 - len(ratings) / (num_users * num_items))
    }
    
    print(f"  Users:                {stats['num_users']}")
    print(f"  Items:                {stats['num_items']}")
    print(f"  Total interactions:   {stats['num_interactions']}")
    print(f"  Avg sequence length:  {stats['avg_seq_len']:.2f}")
    print(f"  Min sequence length:  {stats['min_seq_len']}")
    print(f"  Max sequence length:  {stats['max_seq_len']}")
    print(f"  Sparsity:             {stats['sparsity']:.2f}%")

    
    return train_set, val_set, test_set, stats