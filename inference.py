import torch
import numpy as np
from pathlib import Path
from model import SASRec


class SASRecInference:
    def __init__(self, model_path="results/best_model.pt", device=None):
        """Load trained SASRec model for inference."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.num_items = 3416
        
        # Load model
        self.model = SASRec(
            num_items=self.num_items,
            d_model=50,
            num_blocks=2,
            num_heads=1,
            dropout=0.2,
            max_len=200
        ).to(self.device)
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
    
    def recommend(self, user_history: list, n_recommendations: int = 10, exclude_history: bool = True):
        """
        Get top-N recommendations for a user.
        
        Args:
            user_history: List of item IDs (1-3416)
            n_recommendations: Number of items to recommend
            exclude_history: Whether to exclude already-seen items
        
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        
        if len(user_history) == 0:
            raise ValueError("History cannot be empty")
        
        # Validate item IDs
        invalid = [x for x in user_history if x < 1 or x > self.num_items]
        if invalid:
            raise ValueError(f"Invalid item IDs: {invalid}. Must be in [1, {self.num_items}]")
        
        # Use last 200 items
        if len(user_history) > 200:
            user_history = user_history[-200:]
        
        # Pad to 200
        padded_seq = [0] * (200 - len(user_history)) + user_history
        
        # Get predictions
        with torch.no_grad():
            input_tensor = torch.tensor([padded_seq], dtype=torch.long).to(self.device)
            mask_tensor = torch.tensor(
                [[0.0] * (200 - len(user_history)) + [1.0] * len(user_history)]
            ).to(self.device)
            
            logits = self.model(input_tensor, mask_tensor)
            last_logits = logits[0, -1, :].cpu().numpy()
        
        # Get top-N excluding history
        if exclude_history:
            valid_items = [
                (i, float(last_logits[i])) 
                for i in range(1, self.num_items + 1) 
                if i not in user_history
            ]
        else:
            valid_items = [
                (i, float(last_logits[i])) 
                for i in range(1, self.num_items + 1)
            ]
        
        valid_items.sort(key=lambda x: x[1], reverse=True)
        
        return valid_items[:n_recommendations]


if __name__ == "__main__":
    # Example
    recommender = SASRecInference()
    
    history = [5, 10, 15, 20, 25]
    recommendations = recommender.recommend(history, n_recommendations=10)
    
    print(f"User history: {history}")
    print(f"\nTop 10 recommendations:")
    for rank, (item_id, score) in enumerate(recommendations, 1):
        print(f"  {rank}. Item {item_id:<5} (score: {score:>7.4f})")