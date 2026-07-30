from __future__ import annotations

import math

import torch
from torch import nn


class ProteinCausalLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_len: int,
        dropout: float,
        pad_id: int,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.max_len = int(max_len)
        self.dropout = float(dropout)
        self.pad_id = int(pad_id)

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)
        self.drop = nn.Dropout(self.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layers)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.norm = nn.LayerNorm(self.d_model)

        self.register_buffer("_pos_ids", torch.arange(self.max_len, dtype=torch.long), persistent=False)
        self.num_classes = None

    def set_classification_head(self, num_classes: int):
        self.num_classes = int(num_classes)
        # Classification head: simple pooling + linear
        # We replace lm_head or add a new one? Let's add a new one.
        # But for simplicity, let's reuse the structure or add a specific head.
        self.classifier = nn.Linear(self.d_model, self.num_classes)
        # We don't delete lm_head to avoid breaking loading, but we won't use it.

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # PyTorch Transformer mask: True means prevent attention
        # But for 'mask' arg, it's different.
        # nn.TransformerEncoderLayer expects src_mask to be (S, S) or (B*nhead, S, S).
        # If float, add to attention scores. If bool, True indicates elements NOT allowed to attend.
        # Wait, let's check docs.
        # "If a BoolTensor is provided, positions with True are not allowed to attend while False values will be unchanged."
        # Causal mask: position i can attend to 0..i. So j > i should be masked (True).
        mask = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"expected input_ids as (B,L), got shape={tuple(input_ids.shape)}")
        b, l = input_ids.shape
        if l > self.max_len:
            # truncate
            input_ids = input_ids[:, :self.max_len]
            l = self.max_len
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_len]

        pos = self._pos_ids[:l].unsqueeze(0).expand(b, l)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        src_key_padding_mask = None
        if attention_mask is not None:
            # PyTorch Transformer expects padding mask to be True for padded positions
            # If attention_mask is 1 for valid, 0 for pad
            src_key_padding_mask = (attention_mask == 0)

        if getattr(self, "num_classes", None) is not None:
            mask = None # Bidirectional attention for classification
        else:
            mask = self._causal_mask(l, device=x.device)
            
        h = self.encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        
        if getattr(self, "num_classes", None) is not None:
            # Pooling: mean or cls? Let's use mean of non-padded tokens or just mean.
            if attention_mask is not None:
                # h: (B, L, D), attention_mask: (B, L)
                # mask out padding
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = torch.sum(h * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = h.mean(dim=1)
            
            logits = self.classifier(pooled)
            return logits
        else:
            logits = self.lm_head(h)
            return logits
            
        return self.lm_head(h)


class ProteinBiLSTM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        dropout: float,
        pad_id: int,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.pad_id = int(pad_id)

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.drop = nn.Dropout(self.dropout)

        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.n_layers,
            dropout=self.dropout if self.n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(self.d_model * 2, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def set_classification_head(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        x = self.drop(x)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            # Handle empty sequences if any (shouldn't happen with proper dataset)
            lengths = torch.clamp(lengths, min=1)
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.lstm(x_packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=input_ids.size(1))
        else:
            out, _ = self.lstm(x)

        out = self.proj(out)
        out = self.norm(out)

        if getattr(self, "num_classes", None) is not None:
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = torch.sum(out * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = out.mean(dim=1)
            return self.classifier(pooled)

        return self.lm_head(out)


class ProteinCNN(nn.Module):
    """1D CNN baseline for protein sequences.

    Uses multiple parallel convolutional kernels of different sizes to capture
    multi-scale n-gram patterns (like k-mer motifs), followed by global max
    pooling and a classifier head.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        dropout: float,
        pad_id: int,
        kernel_sizes: list[int] | None = None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.pad_id = int(pad_id)

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7, 9]
        self.kernel_sizes = kernel_sizes

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.drop = nn.Dropout(self.dropout)

        # Multiple parallel conv filters — each captures a different n-gram length
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            # Pad so output length == input length for easy pooling
            conv = nn.Conv1d(
                in_channels=self.d_model,
                out_channels=self.d_model,
                kernel_size=k,
                padding=k // 2,
            )
            self.convs.append(conv)

        # Combined feature dim = d_model * number of kernel sizes
        self.combined_dim = self.d_model * len(self.kernel_sizes)
        self.norm = nn.LayerNorm(self.combined_dim)
        self.lm_head = nn.Linear(self.combined_dim, self.vocab_size, bias=False)
        self.num_classes = None

    def set_classification_head(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(self.combined_dim, self.num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # input_ids: (B, L)
        x = self.tok_emb(input_ids)           # (B, L, D)
        x = self.drop(x)
        x = x.transpose(1, 2)                 # (B, D, L) — Conv1d expects (B, C, L)

        conv_outs = []
        for conv in self.convs:
            out = conv(x)                     # (B, D, L)
            out = torch.relu(out)
            # Global max-pooling over the sequence length
            out = out.max(dim=-1)[0]          # (B, D)
            conv_outs.append(out)

        # Concatenate features from all kernel sizes
        h = torch.cat(conv_outs, dim=-1)      # (B, D * n_kernels)
        h = self.norm(h)

        if getattr(self, "num_classes", None) is not None:
            return self.classifier(h)

        # Project back to vocab for LM mode (less typical for CNN, but compatible)
        return self.lm_head(h)

