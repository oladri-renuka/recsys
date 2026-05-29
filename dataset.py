import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    """Training dataset with negative sampling."""
    
    def __init__(self, data, num_items, max_len=200):
        self.data = data
        self.num_items = num_items
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data[idx]
        user_id = record['user_id']
        input_seq = record['input_seq']
        target_seq = record['target_seq']
        
        # Random negative sampling
        rng = np.random.RandomState(user_id)
        neg_seq = [rng.randint(1, self.num_items + 1) for _ in range(len(target_seq))]
        
        return {
            'user_id': user_id,
            'input_seq': input_seq,
            'target_seq': target_seq,
            'neg_seq': neg_seq
        }


class ValTestDataset(Dataset):
    """Validation/Test dataset."""
    
    def __init__(self, data, max_len=200):
        self.data = data
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data[idx]
        return {
            'user_id': record['user_id'],
            'input_seq': record['input_seq'],
            'target_item': record['target_item']
        }


def collate_train(batch, num_items, max_len):
    """Collate function for training."""
    user_ids = []
    input_ids = []
    target_ids = []
    neg_ids = []
    
    for item in batch:
        user_ids.append(item['user_id'])
        
        input_seq = item['input_seq']
        target_seq = item['target_seq']
        neg_seq = item['neg_seq']
        
        # Left-pad
        pad_len = max_len - len(input_seq)
        input_seq_padded = [0] * pad_len + input_seq
        target_seq_padded = [0] * pad_len + target_seq
        neg_seq_padded = [0] * pad_len + neg_seq
        
        input_ids.append(input_seq_padded)
        target_ids.append(target_seq_padded)
        neg_ids.append(neg_seq_padded)
    
    # Attention mask
    attention_mask = []
    for item in batch:
        seq_len = len(item['input_seq'])
        mask = [0.0] * (max_len - seq_len) + [1.0] * seq_len
        attention_mask.append(mask)
    
    return {
        'user_ids': torch.tensor(user_ids, dtype=torch.long),
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'target_ids': torch.tensor(target_ids, dtype=torch.long),
        'neg_ids': torch.tensor(neg_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.float)
    }


def collate_val_test(batch, max_len):
    """Collate function for validation/test."""
    user_ids = []
    input_ids = []
    target_ids = []
    
    for item in batch:
        user_ids.append(item['user_id'])
        
        input_seq = item['input_seq']
        target_item = item['target_item']
        
        # Left-pad
        pad_len = max_len - len(input_seq)
        input_seq_padded = [0] * pad_len + input_seq
        
        input_ids.append(input_seq_padded)
        target_ids.append(target_item)
    
    # Attention mask
    attention_mask = []
    for item in batch:
        seq_len = len(item['input_seq'])
        mask = [0.0] * (max_len - seq_len) + [1.0] * seq_len
        attention_mask.append(mask)
    
    return {
        'user_ids': torch.tensor(user_ids, dtype=torch.long),
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'target_ids': torch.tensor(target_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.float)
    }


def create_dataloaders(train_set, val_set, test_set, num_items, batch_size=128, max_len=200):
    """Create dataloaders."""
    
    train_dataset = TrainDataset(train_set, num_items, max_len)
    val_dataset = ValTestDataset(val_set, max_len)
    test_dataset = ValTestDataset(test_set, max_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_train(x, num_items, max_len)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_val_test(x, max_len)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_val_test(x, max_len)
    )
    
    return train_loader, val_loader, test_loader