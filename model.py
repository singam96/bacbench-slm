from __future__ import annotations

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

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"expected input_ids as (B,L), got shape={tuple(input_ids.shape)}")
        b, l = input_ids.shape
        if l > self.max_len:
            raise ValueError(f"sequence length {l} exceeds max_len {self.max_len}")

        pos = self._pos_ids[:l].unsqueeze(0).expand(b, l)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        mask = self._causal_mask(l, device=x.device)
        h = self.encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        return self.lm_head(h)
