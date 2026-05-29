import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal mask.
    Fixed: Use -1e10 instead of -inf to avoid NaN in softmax.
    """
    
    def __init__(self, d_model: int, num_heads: int = 1, dropout: float = 0.2):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # FIXED: Use -1e10 instead of -inf
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)) == 1
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e10)
        
        padding_mask = (mask.unsqueeze(1).unsqueeze(1) * mask.unsqueeze(1).unsqueeze(2))
        scores = scores.masked_fill(padding_mask == 0, -1e10)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        output = self.out_proj(attn_output)
        return output


class SASRecBlock(nn.Module):
    """Single SASRec block: Attention + FFN with residuals."""
    
    def __init__(self, d_model: int, num_heads: int = 1, dropout: float = 0.2):
        super().__init__()
        
        # Attention block
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # FFN block - Match the saved model structure
        self.ffn = nn.ModuleDict({
            'linear1': nn.Linear(d_model, d_model),
            'linear2': nn.Linear(d_model, d_model),
            'relu': nn.ReLU()
        })
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        attn_out = self.attention(x, mask)
        attn_out = self.dropout1(attn_out)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn['linear1'](x)
        ffn_out = self.ffn['relu'](ffn_out)
        ffn_out = self.ffn['linear2'](ffn_out)
        ffn_out = self.dropout2(ffn_out)
        x = self.norm2(x + ffn_out)
        
        return x


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model."""
    
    def __init__(
        self,
        num_items: int,
        d_model: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout: float = 0.2,
        max_len: int = 200
    ):
        super().__init__()
        
        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len
        
        # Embeddings
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.positional_embedding = nn.Embedding(max_len, d_model)
        
        # Blocks
        self.blocks = nn.ModuleList([
            SASRecBlock(d_model, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, seq_len, num_items + 1]
        """
        
        seq_len = input_ids.size(1)
        
        # Embeddings
        x = self.item_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device)
        x = x + self.positional_embedding(positions)
        
        # Blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final norm
        x = self.final_norm(x)
        
        # Prediction: logits = h @ W_item.T
        logits = torch.matmul(x, self.item_embedding.weight.T)
        
        return logits