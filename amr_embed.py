"""Genome-level AMR prediction from cached ESM-2 protein embeddings (embed.py).

Every genome is a collection of ~4,700 protein embeddings. Five trainable heads read ALL of them:

  transformer  self-attention over the genome's proteins (treated as a SET; trained on random subsets)
  cnn          1D convolutions (kernels 3/5/7/9) + global max pooling along the genome (genome order)
  bilstm       bidirectional LSTM along the genome (genome order)
  attn         gated multiple-instance attention, one attention map per antibiotic (SET)
  mean         mean over proteins + linear (SET baseline)

Same genomes, split, labels, loss and metrics as amr.py, so results are comparable.
Usage: python amr_embed.py [--modes transformer,cnn,bilstm,attn,mean]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from amr import amr_metrics, load_amr_labels
from config import CONFIG

SEQ_MODES = {"cnn", "bilstm"}            # read proteins in genome order; everything else sees a set


def _proj(d_in: int, hidden: int, dropout: float) -> nn.Module:
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout))


class MILHead(nn.Module):
    """'attn' (gated attention, per-antibiotic) and 'mean' heads."""

    def __init__(self, d_in: int, n_out: int, hidden: int = 128, mode: str = "attn", dropout: float = 0.2):
        super().__init__()
        self.mode = mode
        self.proj = _proj(d_in, hidden, dropout)
        if mode == "attn":
            self.V, self.U = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
            self.w = nn.Linear(hidden, n_out)
        self.out_w = nn.Parameter(torch.randn(n_out, hidden) * hidden ** -0.5)
        self.out_b = nn.Parameter(torch.zeros(n_out))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)                                                        # (B,N,H)
        if self.mode == "attn":
            s = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))        # (B,N,A)
            a = torch.softmax(s.masked_fill(~mask.unsqueeze(-1), float("-inf")), dim=1)
            v = torch.einsum("bna,bnh->bah", a, h)                              # (B,A,H)
        else:
            m = mask.unsqueeze(-1).float()
            v = ((h * m).sum(1) / m.sum(1).clamp(min=1)).unsqueeze(1)           # (B,1,H)
        return (self.drop(v) * self.out_w).sum(-1) + self.out_b


class SeqHead(nn.Module):
    """'transformer', 'cnn' and 'bilstm' heads: contextualise the proteins, then mean+max pool."""

    def __init__(self, d_in: int, n_out: int, mode: str, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.mode = mode
        self.proj = _proj(d_in, hidden, dropout)
        if mode == "transformer":
            layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=2)
            out_dim = hidden
        elif mode == "cnn":
            self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, k, padding=k // 2) for k in (3, 5, 7, 9)])
            out_dim = hidden * 4
        elif mode == "bilstm":
            self.rnn = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
            out_dim = hidden * 2
        else:
            raise ValueError(mode)
        self.norm = nn.LayerNorm(out_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim * 2, n_out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) * mask.unsqueeze(-1)                                   # (B,N,H), pads zeroed
        if self.mode == "transformer":
            h = self.enc(h, src_key_padding_mask=~mask)
        elif self.mode == "cnn":
            t = h.transpose(1, 2)
            h = torch.cat([torch.relu(c(t)) for c in self.convs], dim=1).transpose(1, 2)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(h, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
            h, _ = self.rnn(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        m = mask.unsqueeze(-1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        mx = h.masked_fill(~m, float("-inf")).max(1).values
        return self.fc(self.drop(self.norm(torch.cat([mean, mx], dim=-1))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="/workspace/emb")
    ap.add_argument("--out", default=CONFIG["amr"]["checkpoint_dir"])
    ap.add_argument("--modes", default="transformer,cnn,bilstm,attn,mean")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--protein-dropout", type=float, default=0.2, help="attn/mean heads: fraction of proteins hidden in training")
    ap.add_argument("--set-subset", type=int, default=1024, help="transformer head: proteins sampled per genome in training")
    ap.add_argument("--crop", type=int, default=1536, help="cnn/bilstm heads: contiguous genome window used in training")
    ap.add_argument("--tag", default="esm2_35m")
    ap.add_argument("--resume", action="store_true", help="start from the saved head_<tag>_<mode>.pt weights")
    ap.add_argument("--save-tag", default=None, help="name for outputs (default: --tag); use with --resume to keep originals")
    a = ap.parse_args()
    save_tag = a.save_tag or a.tag
    cfg = CONFIG["amr"]
    os.makedirs(a.out, exist_ok=True)
    seed = int(CONFIG["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda"

    sets = np.load(os.path.join(a.emb, "genomes.npz"))
    names = list(sets["names"])
    src = {"set": (sets["offsets"], torch.from_numpy(sets["flat"].astype(np.int64)))}
    if any(m in SEQ_MODES for m in a.modes.split(",")):
        od = np.load(os.path.join(a.emb, "genome_order.npz"))
        assert list(od["names"]) == names
        src["seq"] = (od["offsets"], torch.from_numpy(od["flat"].astype(np.int64)))
    meta = json.load(open(os.path.join(a.emb, "meta.json")))
    emb = np.memmap(os.path.join(a.emb, "emb.f16"), dtype=np.float16, mode="r", shape=(meta["n_proteins"], meta["dim"]))
    E = torch.from_numpy(np.ascontiguousarray(emb)).to(dev)                   # all proteins live on the GPU
    print(f"{len(names)} genomes, {meta['n_proteins']:,} unique proteins, dim {meta['dim']}", flush=True)

    y, antibiotics = load_amr_labels(cfg, names)
    s = json.load(open(os.path.join(a.out, "split.json")))                    # same split as amr.py
    pos = {n: i for i, n in enumerate(names)}
    train_idx = np.array([pos[n] for n in s["train"]])
    val_idx = np.array([pos[n] for n in s["val"]])
    print(f"train {len(train_idx)}  val {len(val_idx)}  antibiotics {len(antibiotics)}", flush=True)

    ytr = y[train_idx]
    pc, nc = (ytr == 1).sum(0), (ytr == 0).sum(0)
    pos_weight = torch.tensor(np.clip(np.where(pc > 0, nc / np.maximum(pc, 1), 1.0), 1.0, float(cfg["max_pos_weight"])),
                              dtype=torch.float32, device=dev)
    y_t = torch.from_numpy(y).to(dev)

    def batch(ids: np.ndarray, mode: str, train: bool):
        offsets, flat_t = src["seq" if mode in SEQ_MODES else "set"]
        sets_ = []
        for i in ids:
            p = flat_t[offsets[i]:offsets[i + 1]]
            if train and mode == "transformer" and len(p) > a.set_subset:        # random subset of the genome
                p = p[torch.randperm(len(p))[:a.set_subset]]
            elif train and mode in SEQ_MODES and len(p) > a.crop:                # random contiguous window
                st = int(np.random.randint(0, len(p) - a.crop + 1))
                p = p[st:st + a.crop]
            sets_.append(p)
        n = max(len(p) for p in sets_)
        idx = torch.zeros(len(ids), n, dtype=torch.long)
        mask = torch.zeros(len(ids), n, dtype=torch.bool)
        for r, p in enumerate(sets_):
            idx[r, :len(p)] = p
            mask[r, :len(p)] = True
        idx, mask = idx.to(dev), mask.to(dev)
        if train and mode in ("attn", "mean") and a.protein_dropout > 0:
            keep = torch.rand(mask.shape, device=dev) > a.protein_dropout
            keep[:, 0] |= ~keep.any(1)
            mask = mask & keep
        return E[idx].float(), mask, y_t[torch.from_numpy(ids).to(dev)]

    def evaluate(model, mode):
        model.eval()
        sc = []
        with torch.no_grad():
            for i in range(0, len(val_idx), a.eval_batch_size):
                x, m, _ = batch(val_idx[i:i + a.eval_batch_size], mode, False)
                sc.append(torch.sigmoid(model(x, m)).cpu())
        return amr_metrics(torch.cat(sc).numpy(), y[val_idx], antibiotics, int(cfg["min_positives_eval"]))

    summary = {}
    for mode in a.modes.split(","):
        print(f"\n===== head: {mode} =====", flush=True)
        model = (SeqHead(meta["dim"], len(antibiotics), mode) if mode in ("transformer", "cnn", "bilstm")
                 else MILHead(meta["dim"], len(antibiotics), mode=mode)).to(dev)
        print(f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
        best, best_state, bad, t0 = -1.0, None, 0, time.time()
        if a.resume:                                          # continue from the saved best weights
            model.load_state_dict(torch.load(os.path.join(a.out, f"head_{a.tag}_{mode}.pt")))
            rows, macro = evaluate(model, mode)
            best, best_rows, best_macro, best_ep = macro["auprc"], rows, macro, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"resumed: ep  0 val AUPRC {macro['auprc']:.3f} AUROC {macro['auroc']:.3f}", flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        for ep in range(a.epochs):
            model.train()
            perm = np.random.permutation(train_idx)
            tot, nb = 0.0, 0
            for i in range(0, len(perm), a.batch_size):
                x, m, yy = batch(perm[i:i + a.batch_size], mode, True)
                mk = (yy >= 0).float()
                loss = (F.binary_cross_entropy_with_logits(model(x, m), yy.clamp(min=0), pos_weight=pos_weight,
                                                            reduction="none") * mk).sum() / mk.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot, nb = tot + float(loss), nb + 1
            rows, macro = evaluate(model, mode)
            print(f"ep {ep + 1:2d} loss {tot / nb:.4f}  val AUPRC {macro['auprc']:.3f} AUROC {macro['auroc']:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if macro["auprc"] > best:
                best, bad = macro["auprc"], 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_rows, best_macro, best_ep = rows, macro, ep + 1
            else:
                bad += 1
                if bad >= a.patience:
                    print("early stop", flush=True)
                    break
        torch.save(best_state, os.path.join(a.out, f"head_{save_tag}_{mode}.pt"))
        with open(os.path.join(a.out, f"amr_metrics_{save_tag}_{mode}.json"), "w") as f:
            json.dump({"macro": best_macro, "per_antibiotic": best_rows, "epoch": best_ep, "embedding": meta}, f, indent=2)
        summary[f"{save_tag}_{mode}"] = best_macro
        print(f"best epoch {best_ep}: AUPRC {best_macro['auprc']:.3f}  AUROC {best_macro['auroc']:.3f} "
              f"over {best_macro['n_evaluated']} antibiotics  [{time.time() - t0:.0f}s]", flush=True)
        with open(os.path.join(a.out, f"amr_summary_{save_tag}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        del model, opt, best_state


if __name__ == "__main__":
    main()
