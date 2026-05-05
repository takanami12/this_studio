"""
PhaBERT-CNN Model Architecture

Hybrid deep learning framework combining DNABERT-2 contextual embeddings with
multi-scale CNNs, attention pooling, and a hand-crafted bio-feature MLP branch
that injects classical genomic descriptors (k-mer frequencies, GC statistics,
GC-skew Fourier descriptors, dinucleotide odds ratio).

Architecture:
    1. DNABERT-2 backbone (768-dim contextualised embeddings)
    2. Multi-scale CNN branch (3 parallel Conv1d, kernels 3, 5, 7)  -> 384
    3. Attention-based pooling branch                               -> 128
    4. Bio-feature MLP branch (k-mer + GC + GC-skew + dinuc OR)     -> 128
    5. Classification head: concat (640) -> binary
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:    (batch_size, embedding_dim, seq_len) - Transposed DNABERT-2 embeddings
            mask: (batch_size, seq_len) - Attention mask (1 = real, 0 = PAD).
                  When provided, PAD positions are forced to -inf before the
                  AdaptiveMaxPool1d so they cannot be selected. Same-padding
                  Conv1d preserves seq_len, so the mask aligns 1-to-1 with the
                  feature map.
        Returns:
            out: (batch_size, 128) - Pooled feature vector
        """
        out = self.conv_block(x)     # (B, 128, L)
        if mask is not None:
            # (B, L) -> (B, 1, L) so it broadcasts over channels.
            mask_expanded = mask.unsqueeze(1).to(out.dtype)
            out = out.masked_fill(mask_expanded == 0, float("-inf"))
        out = self.pool(out)          # (B, 128, 1)
        out = out.squeeze(-1)         # (B, 128)
        return out


class BioFeatureMLP(nn.Module):
    """MLP branch that consumes normalised hand-crafted bio-features.

    Two hidden layers with GELU + dropout. LayerNorm at the input keeps the
    branch stable when fed raw z-scored features whose scale may still vary
    across columns (e.g. odds-ratio entries vs k-mer frequencies).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhaBERTCNN(nn.Module):
    """
    PhaBERT-CNN with bio-feature fusion.

    Components:
        1. DNABERT-2 backbone
        2. Multi-scale CNN on contextual embeddings (3 branches)  -> 384
        3. Attention pooling                                       -> 128
        4. Bio-feature MLP                                         -> 128
        5. Classification head: concat (640) -> binary
    """

    def __init__(
        self,
        dnabert2_model_name: str = "zhihan1996/DNABERT-2-117M",
        embedding_dim: int = 768,
        cnn_kernel_sizes: list = [3, 5, 7],
        attention_hidden_dim: int = 64,
        attention_dropout: float = 0.1,
        bio_feature_dim: int = 0,
        bio_branch_hidden_dim: int = 256,
        bio_branch_output_dim: int = 128,
        bio_branch_dropout: float = 0.2,
        classifier_hidden_dim: int = 256,
        classifier_dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.bio_feature_dim = bio_feature_dim

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
        # Pre-load the BertEncoder class from the same dynamic module and
        # patch `rebuild_alibi_tensor` to materialise on CPU.  transformers
        # >= 5.x (and torch >= 2.10) push HF `from_pretrained` through a
        # meta-device default; DNABERT-2's original ALiBi rebuild then mixes
        # meta + cpu tensors and crashes during __init__.
        encoder_cls = get_class_from_dynamic_module(
            "bert_layers.BertEncoder",
            dnabert2_model_name,
            trust_remote_code=True,
        )
        if not getattr(encoder_cls, "_alibi_cpu_patched", False):
            _orig_rebuild = encoder_cls.rebuild_alibi_tensor

            def _rebuild_alibi_cpu(self, size, device=None):
                return _orig_rebuild(self, size=size,
                                     device=device or torch.device("cpu"))

            encoder_cls.rebuild_alibi_tensor = _rebuild_alibi_cpu
            encoder_cls._alibi_cpu_patched = True

        with torch.device("cpu"):
            self.backbone = model_cls.from_pretrained(
                dnabert2_model_name,
                config=config,
                low_cpu_mem_usage=False,
            )

        # The DNABERT-2 self-attention layer hard-imports a Triton flash
        # kernel. The shipped flash_attn_triton.py uses `tl.dot(..., trans_b=True)`,
        # which was removed in Triton >= 3.x — kernel compilation crashes on
        # any modern CUDA stack. Disable unconditionally so the layer falls
        # back to the pure-PyTorch attention branch
        # (see `bert_layers.BertUnpadSelfAttention.forward`, line ~161).
        import sys
        backbone_module = sys.modules.get(model_cls.__module__)
        if backbone_module is not None and hasattr(
            backbone_module, "flash_attn_qkvpacked_func"
        ):
            backbone_module.flash_attn_qkvpacked_func = None

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
        # Component 4: Hand-crafted Bio-Feature Branch (Phase 1 enrichment)
        # Disabled when bio_feature_dim == 0 to allow ablation.
        # ============================================================
        if bio_feature_dim > 0:
            self.bio_branch = BioFeatureMLP(
                input_dim=bio_feature_dim,
                hidden_dim=bio_branch_hidden_dim,
                output_dim=bio_branch_output_dim,
                dropout=bio_branch_dropout,
            )
            bio_output_dim = bio_branch_output_dim
        else:
            self.bio_branch = None
            bio_output_dim = 0

        # ============================================================
        # Component 5: Classification Head
        # Input: 384 (CNN) + 128 (attention) + 128 (bio) = 640
        # ============================================================
        total_feature_dim = cnn_output_dim + 128 + bio_output_dim

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
        bio_features: torch.Tensor = None,
    ):
        """
        Forward pass of PhaBERT-CNN.

        Args:
            input_ids: (batch_size, seq_len) - Tokenized DNA sequences
            attention_mask: (batch_size, seq_len) - Attention mask
            bio_features: (batch_size, bio_feature_dim) - Pre-normalised
                hand-crafted descriptors. Required when the bio branch is
                enabled (bio_feature_dim > 0).

        Returns:
            logits: (batch_size, num_classes)
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
        # Zero out PAD positions before Conv so conv outputs at boundary
        # positions are not contaminated by [PAD] embeddings, then mask the
        # post-conv feature map again (handled inside each branch's pool).
        if attention_mask is not None:
            hidden_states_for_cnn = hidden_states * attention_mask.unsqueeze(-1).to(
                hidden_states.dtype
            )
        else:
            hidden_states_for_cnn = hidden_states
        # Transpose for Conv1d: (B, L, 768) -> (B, 768, L)
        cnn_input = hidden_states_for_cnn.transpose(1, 2)

        cnn_features = []
        for branch in self.cnn_branches:
            feat = branch(cnn_input, mask=attention_mask)  # (B, 128)
            cnn_features.append(feat)

        # Concatenate CNN features: (B, 384)
        cnn_out = torch.cat(cnn_features, dim=-1)

        # Step 3: Attention-based pooling branch
        context_vector, attn_weights = self.attention_pooling(
            hidden_states, attention_mask
        )
        # Project: 768 -> 128
        global_out = self.global_projection(context_vector)  # (B, 128)

        # Step 4: Bio-feature branch (optional)
        feature_parts = [cnn_out, global_out]
        if self.bio_branch is not None:
            if bio_features is None:
                raise ValueError(
                    "bio_features must be provided when bio_feature_dim > 0"
                )
            bio_out = self.bio_branch(bio_features.to(cnn_out.dtype))
            feature_parts.append(bio_out)

        combined = torch.cat(feature_parts, dim=-1)

        logits = self.classifier(combined)
        return logits

    def get_backbone_params(self):
        """Get DNABERT-2 backbone parameters (for discriminative LR)."""
        return self.backbone.parameters()

    def get_task_params(self):
        """Get task-specific layer parameters (CNN, attention, bio, classifier)."""
        task_modules = [
            self.cnn_branches,
            self.attention_pooling,
            self.global_projection,
            self.classifier,
        ]
        if self.bio_branch is not None:
            task_modules.append(self.bio_branch)
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
