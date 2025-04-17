import math
from typing import Optional, Tuple

import torch
from torch import nn

# Refer to the LlamaAttention implementation in Hugging Face's modeling_llama.py.
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module to compute the attention weights and output.
        
    Args:
        hidden_size (int): The hidden size of the input tensors (query, key, and value in self-attention is same size).
        num_heads (int): The number of attention heads for the query projections.
        num_key_value_heads (int): The number of attention heads for the key and value projections.
        dropout (float): The dropout probability for the attention weights.
        bias (bool): Whether to use bias in the projection layer.
    """
    def __init__(
        self, 
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        dropout: float,
        bias: bool,
    ):
        super().__init__()
        # Initialize the settings
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.dropout = dropout
        self.bias = bias
        
        # Initialize the projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Get the size of the input tensors
        bsz, q_len, _ = query.size()
        bsz, k_len, _ = key.size()
        
        # Project the query, key, and value into hidden states
        query_states = self.q_proj(query)  # Shape: (batch_size, q_len, num_heads * head_dim)
        key_states = self.k_proj(key)  # Shape: (batch_size, k_len, num_heads * head_dim)
        value_states = self.v_proj(value)  # Shape: (batch_size, k_len, num_heads * head_dim)

        # Reshape the query, key, and value hidden states into multi-heads
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) # Shape: (batch_size, num_heads, q_len, head_dim)
        key_states = key_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # Shape: (batch_size, num_key_value_heads, k_len, head_dim)
        value_states = value_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # Shape: (batch_size, num_key_value_heads, k_len, head_dim)
        
        # Compute the attention weights
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim) # Shape: (batch_size, num_heads, q_len, k_len)
        
        # Mask the attention weights with the key padding mask
        attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf")) # Shape: (batch_size, num_heads, q_len, k_len)
        
        # Normalize the attention weights
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) # Shape: (batch_size, num_heads, q_len, k_len)
        
        # Dropout the attention weights if training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training) # Shape: (batch_size, num_heads, q_len, k_len)
        
        # Compute the attention output
        attn_output = torch.matmul(attn_weights, value_states) # Shape: (batch_size, num_heads, q_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous() # Shape: (batch_size, q_len, num_heads, head_dim)
        attn_output = attn_output.view(bsz, q_len, -1) # Shape: (batch_size, q_len, hidden_size)
        
        return attn_output
