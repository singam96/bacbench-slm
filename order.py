"""Per-genome protein-id sequences in GENOME ORDER (contigs in order, genes in order, duplicates kept).

embed.py stores each genome as a de-duplicated SET (genomes.npz). The CNN / BiLSTM heads in
amr_embed.py read proteins along the genome, so this writes genome_order.npz with the same protein
ids but in their natural order. Does not touch the embeddings.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

from config import CONFIG


def main() -> None:
    import datasets
    from tqdm import tqdm

    from amr import load_amr_labels
    from dataset import _flatten_strings

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/emb")
    a = ap.parse_args()
    with open(os.path.join(a.out, "seqs.pkl"), "rb") as f:
        uniq = {s: i for i, s in enumerate(pickle.load(f))}

    ds = datasets.load_dataset(CONFIG["dataset_name"], split=CONFIG["dataset_split"])
    target = str(CONFIG["species_filter"])
    ds = ds.select([i for i, t in enumerate(ds[CONFIG["species_column"]]) if str(t) == target])
    names = list(ds["genome_name"])
    y, _ = load_amr_labels(CONFIG["amr"], names)
    rows = np.where((y >= 0).any(axis=1))[0]
    g = np.load(os.path.join(a.out, "genomes.npz"))
    assert list(g["names"]) == [names[r] for r in rows], "genome selection differs from embed.py"

    flat = []
    for r in tqdm(rows, desc="ordering"):
        ids = [uniq[s.strip().upper()] for s in _flatten_strings(ds[int(r)][CONFIG["protein_column"]])
               if isinstance(s, str) and s.strip()]
        flat.append(np.asarray(ids, dtype=np.int32))
    offsets = np.zeros(len(flat) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(f) for f in flat])
    np.savez(os.path.join(a.out, "genome_order.npz"), names=g["names"], offsets=offsets, flat=np.concatenate(flat))
    print(f"wrote genome_order.npz: {len(flat)} genomes, {offsets[-1]:,} protein occurrences, "
          f"longest genome {max(len(f) for f in flat)}")


if __name__ == "__main__":
    main()
