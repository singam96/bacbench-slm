from __future__ import annotations

import ast
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
        return
    if isinstance(x, str):
        s = x.strip()
        # Handle stringified lists e.g. "['a', 'b']"
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    for item in parsed:
                        yield from _flatten_strings(item)
                    return
            except (ValueError, SyntaxError):
                pass
        yield x
        return
    if isinstance(x, (list, tuple, set)):
        for item in x:
            yield from _flatten_strings(item)
        return


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
        max_proteins_per_genome: int | None = None,
        seed: int = 0,
        label_column: str | None = None,
        label_map: dict[str, int] | None = None,
    ):
        self.hf_dataset = hf_dataset
        self.genome_indices = list(genome_indices)
        self.protein_column = str(protein_column)
        self.vocab = vocab
        self.max_len = int(max_len)
        self.min_seq_len = int(min_seq_len)
        self.max_proteins_per_genome = None if max_proteins_per_genome is None else int(max_proteins_per_genome)
        self.seed = int(seed)
        self.label_column = label_column
        self.label_map = label_map

        if self.max_len <= 4:
            raise ValueError(f"max_len must be > 4, got {self.max_len}")
        if self.min_seq_len < 0:
            raise ValueError(f"min_seq_len must be >= 0, got {self.min_seq_len}")
        if self.max_proteins_per_genome is not None and self.max_proteins_per_genome <= 0:
            raise ValueError(
                f"max_proteins_per_genome must be > 0 when set, got {self.max_proteins_per_genome}"
            )

        self._valid_positions: list[list[int]] = []
        self._counts: list[int] = []
        self._cum_counts: list[int] = []
        total = 0
        for row_pos, gi in enumerate(self.genome_indices):
            row = self.hf_dataset[int(gi)]
            seqs = list(_flatten_strings(row.get(self.protein_column)))
            valid = [i for i, s in enumerate(seqs) if isinstance(s, str) and len(s.strip()) >= self.min_seq_len]

            if self.max_proteins_per_genome is not None and len(valid) > self.max_proteins_per_genome:
                g = torch.Generator().manual_seed(self.seed + int(row_pos) * 1_000_003)
                perm = torch.randperm(len(valid), generator=g).tolist()
                valid = [valid[i] for i in perm[: self.max_proteins_per_genome]]

            self._valid_positions.append(valid)
            self._counts.append(len(valid))
            total += len(valid)
            self._cum_counts.append(total)

    def __len__(self) -> int:
        return self._cum_counts[-1] if self._cum_counts else 0

    def _find_genome_and_protein_idx(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        
        # Find which genome this index belongs to
        row_pos = 0
        if self._cum_counts:
            import bisect
            row_pos = bisect.bisect_right(self._cum_counts, int(idx))
            if row_pos >= len(self._cum_counts):
                # Handle edge case where bisect returns len if idx is out of bounds 
                # (though the check above should prevent this)
                row_pos = len(self._cum_counts) - 1
            
        prev_count = self._cum_counts[row_pos - 1] if row_pos > 0 else 0
        offset = int(idx) - int(prev_count)
        
        # Get genome index from our filtered list
        genome_idx = self.genome_indices[row_pos]
        
        # Get the specific protein index within this genome
        # self._valid_positions stores the INDICES of valid proteins in the original list
        protein_idx_in_genome = self._valid_positions[row_pos][offset]
        
        return int(genome_idx), int(protein_idx_in_genome)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        genome_idx, protein_idx = self._find_genome_and_protein_idx(idx)
        genome = self.hf_dataset[genome_idx]
        
        # Get sequence
        seq_list = list(_flatten_strings(genome[self.protein_column]))
        seq = seq_list[protein_idx]
        tokens = tokenize_protein_sequence(seq, vocab=self.vocab, max_len=self.max_len)
        attention_mask = (tokens != self.vocab.pad_id).long()
        
        # Base item (Causal LM style)
        # Note: input_ids for classification is usually the whole sequence.
        # For Causal LM, it is [:-1] and labels [1:]
        
        item = {
            "input_ids": tokens[:-1], # Default to LM inputs
            "attention_mask": attention_mask[:-1],
            "labels": tokens[1:], # For causal LM
        }

        # Add Classification Label if configured
        if self.label_column and self.label_map:
            label_list = list(_flatten_strings(genome[self.label_column]))
            if protein_idx < len(label_list):
                raw_label = label_list[protein_idx]
                # In test mode/limited labels, raw_label might not be in map.
                # If not found, use -100 (ignore) or <unk> if present?
                # The label_map usually has <unk> if we configured it, but let's check.
                label_id = self.label_map.get(str(raw_label).strip(), -100) 
            else:
                label_id = -100
            
            # OVERWRITE labels for classification task.
            # Use full sequence for input_ids if classification? 
            # Actually, standard causal LM input is fine, but maybe we want full sequence.
            # tokenize_protein_sequence returns full sequence with BOS/EOS/PAD.
            # If we slice [:-1], we lose the last token (EOS or PAD).
            # For classification, we usually want [BOS, ..., EOS, PAD].
            # So let's use full tokens for input_ids if classification.
            item["input_ids"] = tokens
            item["attention_mask"] = attention_mask
            item["labels"] = torch.tensor(label_id, dtype=torch.long)

        return item
