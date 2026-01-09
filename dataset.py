from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class ProteinVocab:
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @property
    def size(self) -> int:
        return len(self.id_to_token)


def build_protein_vocab() -> ProteinVocab:
    id_to_token = ["<pad>", "<bos>", "<eos>", "<unk>"] + list(AMINO_ACIDS)
    token_to_id = {t: i for i, t in enumerate(id_to_token)}
    return ProteinVocab(
        pad_id=token_to_id["<pad>"],
        bos_id=token_to_id["<bos>"],
        eos_id=token_to_id["<eos>"],
        unk_id=token_to_id["<unk>"],
        token_to_id=token_to_id,
        id_to_token=id_to_token,
    )


def _flatten_strings(x: Any) -> Iterable[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        out: list[str] = []
        for item in x:
            out.extend(list(_flatten_strings(item)))
        return out
    return []


def tokenize_protein_sequence(
    seq: str,
    *,
    vocab: ProteinVocab,
    max_len: int,
) -> torch.Tensor:
    s = (seq or "").strip().upper().replace(" ", "").replace("\n", "")
    ids = [vocab.bos_id]
    for ch in s:
        ids.append(vocab.token_to_id.get(ch, vocab.unk_id))
        if len(ids) >= max_len - 1:
            break
    ids.append(vocab.eos_id)
    if len(ids) < max_len:
        ids.extend([vocab.pad_id] * (max_len - len(ids)))
    else:
        ids = ids[:max_len]
        ids[-1] = vocab.eos_id
    return torch.tensor(ids, dtype=torch.long)


class GenomeProteinIndexDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        genome_indices: list[int],
        *,
        protein_column: str,
        vocab: ProteinVocab,
        max_len: int,
        min_seq_len: int,
    ):
        self.hf_dataset = hf_dataset
        self.genome_indices = list(genome_indices)
        self.protein_column = str(protein_column)
        self.vocab = vocab
        self.max_len = int(max_len)
        self.min_seq_len = int(min_seq_len)

        if self.max_len <= 4:
            raise ValueError(f"max_len must be > 4, got {self.max_len}")
        if self.min_seq_len < 0:
            raise ValueError(f"min_seq_len must be >= 0, got {self.min_seq_len}")

        self._counts: list[int] = []
        self._cum_counts: list[int] = []
        total = 0
        for gi in self.genome_indices:
            row = self.hf_dataset[int(gi)]
            seqs = list(_flatten_strings(row.get(self.protein_column)))
            n = 0
            for s in seqs:
                if isinstance(s, str) and len(s.strip()) >= self.min_seq_len:
                    n += 1
            self._counts.append(n)
            total += n
            self._cum_counts.append(total)

    def __len__(self) -> int:
        return self._cum_counts[-1] if self._cum_counts else 0

    def _get_nth_valid_sequence(self, genome_row: dict[str, Any], n: int) -> str:
        seqs = list(_flatten_strings(genome_row.get(self.protein_column)))
        k = 0
        for s in seqs:
            if not isinstance(s, str):
                continue
            if len(s.strip()) < self.min_seq_len:
                continue
            if k == n:
                return s
            k += 1
        raise IndexError(f"sequence offset out of range: {n}")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        row_pos = bisect_right(self._cum_counts, int(idx))
        prev = self._cum_counts[row_pos - 1] if row_pos > 0 else 0
        offset = int(idx) - int(prev)
        genome_idx = int(self.genome_indices[row_pos])
        row = self.hf_dataset[genome_idx]
        seq = self._get_nth_valid_sequence(row, offset)
        input_ids = tokenize_protein_sequence(seq, vocab=self.vocab, max_len=self.max_len)
        attention_mask = (input_ids != int(self.vocab.pad_id)).to(torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
