"""Embed every unique protein of the labelled E. coli genomes with a pretrained ESM-2 model.

Stage 1 (count): walk all genomes, de-duplicate identical protein sequences, and record for each
                 genome the ids of the unique proteins it carries.
Stage 2 (embed): run ESM-2 (fp16) once per unique sequence, mean-pool residues, and store the
                 vectors in a float16 memmap. Resumable.

Outputs (in --out, default /workspace/emb):
    genomes.npz   names, offsets, flat   (genome i owns flat[offsets[i]:offsets[i+1]] = protein ids)
    seqs.pkl      list[str] indexed by protein id
    emb.f16       float16 [n_proteins, dim] memmap
    meta.json     model name, dim, n_proteins

Usage: python embed.py [--model facebook/esm2_t12_35M_UR50D] [--stage all|count|embed]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch

from config import CONFIG


def stage_count(out: str) -> None:
    import datasets
    from tqdm import tqdm

    from amr import load_amr_labels
    from dataset import _flatten_strings

    ds = datasets.load_dataset(CONFIG["dataset_name"], split=CONFIG["dataset_split"])
    target = str(CONFIG["species_filter"])
    keep = [i for i, t in enumerate(ds[CONFIG["species_column"]]) if str(t) == target]
    ds = ds.select(keep)
    names = list(ds["genome_name"])
    y, _ = load_amr_labels(CONFIG["amr"], names)
    rows = np.where((y >= 0).any(axis=1))[0]                 # same selection / order as amr.py
    print(f"{len(rows)} labelled genomes of {len(ds)}", flush=True)

    uniq: dict[str, int] = {}
    flat: list[np.ndarray] = []
    total = 0
    t0 = time.time()
    for r in tqdm(rows, desc="de-duplicating"):
        ids = []
        for s in _flatten_strings(ds[int(r)][CONFIG["protein_column"]]):
            s = s.strip().upper() if isinstance(s, str) else ""
            if not s:
                continue
            k = uniq.get(s)
            if k is None:
                k = len(uniq)
                uniq[s] = k
            ids.append(k)
        total += len(ids)
        flat.append(np.unique(np.asarray(ids, dtype=np.int32)))   # a genome is a SET of proteins
    offsets = np.zeros(len(flat) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(f) for f in flat])
    seqs = [None] * len(uniq)
    for s, k in uniq.items():
        seqs[k] = s
    lens = np.array([len(s) for s in seqs])
    print(f"protein occurrences {total:,}  unique sequences {len(seqs):,} "
          f"({100 * len(seqs) / max(total, 1):.1f}% of occurrences)  "
          f"mean len {lens.mean():.0f}  >512 aa: {(lens > 512).mean() * 100:.1f}%  "
          f"mean proteins/genome (unique) {offsets[-1] / len(flat):.0f}  [{time.time() - t0:.0f}s]", flush=True)

    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "genomes.npz"), names=np.array([names[r] for r in rows]),
             offsets=offsets, flat=np.concatenate(flat))
    with open(os.path.join(out, "seqs.pkl"), "wb") as f:
        pickle.dump(seqs, f, protocol=4)


def stage_embed(out: str, model_name: str, max_len: int, token_budget: int) -> None:
    from transformers import AutoTokenizer, EsmModel

    with open(os.path.join(out, "seqs.pkl"), "rb") as f:
        seqs = pickle.load(f)
    n = len(seqs)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()
    dim = int(model.config.hidden_size)
    pad_id, cls_id, eos_id, unk_id = tok.pad_token_id, tok.cls_token_id, tok.eos_token_id, tok.unk_token_id

    lut = np.full(256, unk_id, dtype=np.int64)                # fast residue -> token id table
    for t, i in tok.get_vocab().items():
        if len(t) == 1:
            lut[ord(t)] = i

    order = np.argsort(np.array([min(len(s), max_len) for s in seqs]), kind="stable")
    batches, cur, cur_max = [], [], 0
    for idx in order:
        L = min(len(seqs[idx]), max_len) + 2
        if cur and max(cur_max, L) * (len(cur) + 1) > token_budget:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(int(idx))
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)

    path = os.path.join(out, "emb.f16")
    prog_path = os.path.join(out, "embed_progress.json")
    done = json.load(open(prog_path))["batches_done"] if os.path.exists(prog_path) else 0
    mm = np.memmap(path, dtype=np.float16, mode="r+" if os.path.exists(path) else "w+", shape=(n, dim))
    print(f"embedding {n:,} proteins in {len(batches):,} batches with {model_name} (dim {dim}); resuming at batch {done}",
          flush=True)

    t0, seen = time.time(), 0
    with torch.inference_mode():
        for b in range(done, len(batches)):
            ids = batches[b]
            Lmax = max(min(len(seqs[i]), max_len) for i in ids) + 2
            x = np.full((len(ids), Lmax), pad_id, dtype=np.int64)
            for r, i in enumerate(ids):
                body = lut[np.frombuffer(seqs[i][:max_len].encode("ascii", "ignore"), dtype=np.uint8)]
                x[r, 0] = cls_id
                x[r, 1:1 + len(body)] = body
                x[r, 1 + len(body)] = eos_id
            x = torch.from_numpy(x).cuda(non_blocking=True)
            attn = (x != pad_id)
            h = model(input_ids=x, attention_mask=attn.long()).last_hidden_state.float()
            m = attn.clone()
            m[:, 0] = False                                      # drop <cls>
            m[torch.arange(len(ids), device=m.device), attn.sum(1) - 1] = False   # drop <eos>
            m = m.unsqueeze(-1).float()
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
            mm[np.array(ids)] = emb.half().cpu().numpy()
            seen += len(ids)
            if (b + 1) % 200 == 0 or b == len(batches) - 1:
                mm.flush()
                json.dump({"batches_done": b + 1}, open(prog_path, "w"))
                rate = seen / (time.time() - t0)
                print(f"batch {b + 1}/{len(batches)}  {rate:.0f} seq/s  "
                      f"ETA {(n - done_count(batches, b + 1)) / max(rate, 1) / 60:.1f} min", flush=True)
    mm.flush()
    json.dump({"model": model_name, "dim": dim, "n_proteins": n}, open(os.path.join(out, "meta.json"), "w"))
    print("embedding complete", flush=True)


def done_count(batches, upto):
    return sum(len(b) for b in batches[:upto])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--stage", choices=["all", "count", "embed"], default="all")
    ap.add_argument("--out", default="/workspace/emb")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--token-budget", type=int, default=32768)
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(".cache", "huggingface")))
    if a.stage in ("all", "count"):
        stage_count(a.out)
    if a.stage in ("all", "embed"):
        stage_embed(a.out, a.model, a.max_len, a.token_budget)
