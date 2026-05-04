"""
PhaBERT-CNN Model Architecture

A hybrid deep learning framework integrating DNABERT-2 pre-trained
foundation model with multi-scale CNN and attention-based pooling
for bacteriophage lifestyle classification.

Architecture:
    1. DNABERT-2 backbone (768-dim contextualized embeddings)
    2. Multi-scale CNN branch (3 parallel Conv1d with kernel sizes 3, 5, 7)
    3. Attention-based pooling branch (global sequence representation)
    4. Classification head (512-dim -> binary output)
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .attention import AttentionPooling


class MultiScaleCNNBranch(nn.Module):
    """
    Single CNN branch with two stacked Conv1d layers.
    
    Conv1d(768 -> 256, kernel_size=k) -> BN -> ReLU ->
    Conv1d(256 -> 128, kernel_size=k) -> BN -> ReLU ->
    AdaptiveMaxPool1d(1)
    """
    
    def __init__(self, in_channels: int, kernel_size: int):
        super().__init__()
        
        padding1 = kernel_size // 2
        padding2 = kernel_size // 2
        
        self.conv_block = nn.Sequential(
            # First conv layer: 768 -> 256
            nn.Conv1d(in_channels, 256, kernel_size=kernel_size, padding=padding1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            # Second conv layer: 256 -> 128
            nn.Conv1d(256, 128, kernel_size=kernel_size, padding=padding2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        
        # Adaptive max pooling to fixed-size output
        self.pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, embedding_dim, seq_len) - Transposed DNABERT-2 embeddings
        Returns:
            out: (batch_size, 128) - Pooled feature vector
        """
        out = self.conv_block(x)     # (B, 128, L')
        out = self.pool(out)          # (B, 128, 1)
        out = out.squeeze(-1)         # (B, 128)
        return out


class PhaBERTCNN(nn.Module):
    """
    PhaBERT-CNN: DNABERT-2 + Multi-scale CNN + Attention Pooling
    
    Components:
        1. DNABERT-2 backbone: Pre-trained genome foundation model
        2. Multi-scale CNN: 3 parallel branches (kernel sizes 3, 5, 7)
           Each outputs 128-dim feature vector -> total 384-dim
        3. Attention pooling: Global sequence representation -> 128-dim
        4. Classification head: 512-dim -> binary (virulent vs temperate)
    """
    
    def __init__(
        self,
        dnabert2_model_name: str = "zhihan1996/DNABERT-2-117M",
        embedding_dim: int = 768,
        cnn_kernel_sizes: list = [3, 5, 7],
        attention_hidden_dim: int = 64,
        attention_dropout: float = 0.1,
        classifier_hidden_dim: int = 256,
        classifier_dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # ============================================================
        # Component 1: DNABERT-2 Backbone
        # ============================================================
        from transformers import BertConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        
        config = BertConfig.from_pretrained(dnabert2_model_name)
        if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
            config.pad_token_id = 0
        
        # Disable flash attention (incompatible with newer Triton versions)
        config.use_flash_attn = False
        
        model_cls = get_class_from_dynamic_module(
            "bert_layers.BertModel",
            dnabert2_model_name,
            trust_remote_code=True,
        )
        # Force CPU init to avoid `meta` device mismatch in DNABERT-2's
        # rebuild_alibi_tensor (transformers >=4.40 inits on meta by default).
        with torch.device("cpu"):
            self.backbone = model_cls.from_pretrained(
                dnabert2_model_name,
                config=config,
                low_cpu_mem_usage=False,
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            dnabert2_model_name,
            trust_remote_code=True,
        )
        
        # ============================================================
        # Component 2: Multi-scale CNN Feature Extractor
        # Three parallel convolutional pathways with kernel sizes 3, 5, 7
        # ============================================================
        self.cnn_branches = nn.ModuleList([
            MultiScaleCNNBranch(embedding_dim, ks) for ks in cnn_kernel_sizes
        ])
        cnn_output_dim = 128 * len(cnn_kernel_sizes)  # 128 * 3 = 384
        
        # ============================================================
        # Component 3: Attention-based Pooling (Global Feature Extractor)
        # ============================================================
        self.attention_pooling = AttentionPooling(
            embedding_dim=embedding_dim,
            hidden_dim=attention_hidden_dim,
        )
        
        # Linear projection: 768 -> 128 with ReLU and dropout
        self.global_projection = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(attention_dropout),
        )
        
        # ============================================================
        # Component 4: Classification Head
        # Input: 384 (CNN) + 128 (attention) = 512
        # ============================================================
        total_feature_dim = cnn_output_dim + 128  # 512
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_feature_dim),
            nn.Linear(total_feature_dim, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        """
        Forward pass of PhaBERT-CNN.
        
        Args:
            input_ids: (batch_size, seq_len) - Tokenized DNA sequences
            attention_mask: (batch_size, seq_len) - Attention mask
            
        Returns:
            logits: (batch_size, num_classes) - Classification logits
        """
        # Step 1: Get DNABERT-2 embeddings
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # hidden_states: (B, L, 768)
        # DNABERT-2 custom model returns tuple, not ModelOutput
        if isinstance(backbone_outputs, tuple):
            hidden_states = backbone_outputs[0]
        else:
            hidden_states = backbone_outputs.last_hidden_state
        
        # Step 2: Multi-scale CNN branch
        # Transpose for Conv1d: (B, L, 768) -> (B, 768, L)
        cnn_input = hidden_states.transpose(1, 2)
        
        cnn_features = []
        for branch in self.cnn_branches:
            feat = branch(cnn_input)  # (B, 128)
            cnn_features.append(feat)
        
        # Concatenate CNN features: (B, 384)
        cnn_out = torch.cat(cnn_features, dim=-1)
        
        # Step 3: Attention-based pooling branch
        context_vector, attn_weights = self.attention_pooling(
            hidden_states, attention_mask
        )
        # Project: 768 -> 128
        global_out = self.global_projection(context_vector)  # (B, 128)
        
        # Step 4: Concatenate all features: (B, 512)
        combined = torch.cat([cnn_out, global_out], dim=-1)
        
        # Step 5: Classification
        logits = self.classifier(combined)  # (B, 2)
        
        return logits
    
    def get_backbone_params(self):
        """Get DNABERT-2 backbone parameters (for discriminative LR)."""
        return self.backbone.parameters()
    
    def get_task_params(self):
        """Get task-specific layer parameters (CNN, attention, classifier)."""
        task_modules = [
            self.cnn_branches,
            self.attention_pooling,
            self.global_projection,
            self.classifier,
        ]
        params = []
        for module in task_modules:
            params.extend(module.parameters())
        return params

    def freeze_backbone(self):
        """Freeze DNABERT-2 backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze DNABERT-2 backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
