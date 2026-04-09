"""
Attention-based pooling module for PhaBERT-CNN.

Implements the self-attention mechanism described in Lin et al. (2017)
"A Structured Self-Attentive Sentence Embedding" for aggregating
sequence-level information from DNABERT-2 transformer embeddings.

Given hidden states h_t at position t:
  u_t = tanh(W_a * h_t + b_a)
  alpha_t = softmax(u_t^T * u_s)
  v = sum(alpha_t * h_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Self-attention pooling that learns to weight the importance of
    different sequence positions dynamically.
    
    Projects 768-dim DNABERT-2 embeddings through:
    1. Linear(768 -> 64) with Tanh activation
    2. Linear(64 -> 1) for scalar attention scores  
    3. Softmax normalization across sequence dimension
    4. Weighted sum to produce 768-dim sequence representation
    """
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 64):
        super().__init__()
        
        # Feed-forward network for computing attention scores
        self.W_a = nn.Linear(embedding_dim, hidden_dim)
        self.u_s = nn.Linear(hidden_dim, 1, bias=False)  # Context vector
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            hidden_states: (batch_size, seq_len, embedding_dim) - DNABERT-2 outputs
            attention_mask: (batch_size, seq_len) - 1 for real tokens, 0 for padding
            
        Returns:
            context_vector: (batch_size, embedding_dim) - Weighted sequence representation
            attention_weights: (batch_size, seq_len) - Attention distribution
        """
        # u_t = tanh(W_a * h_t + b_a)
        u = torch.tanh(self.W_a(hidden_states))  # (B, L, hidden_dim)
        
        # Scalar attention scores
        scores = self.u_s(u).squeeze(-1)  # (B, L)
        
        # Mask padding tokens before softmax
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # alpha_t = softmax(scores)
        attention_weights = F.softmax(scores, dim=-1)  # (B, L)
        
        # v = sum(alpha_t * h_t)
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # (B, 1, L)
            hidden_states                     # (B, L, D)
        ).squeeze(1)  # (B, D)
        
        return context_vector, attention_weights
