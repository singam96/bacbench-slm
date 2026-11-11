"""Genome-level antimicrobial-resistance (AMR) prediction.

One model predicts resistant / susceptible for every antibiotic in the label table
(multi-label, multi-task) from the proteins of a genome:

    proteins --(Transformer | BiLSTM | CNN encoder)--> protein embeddings
             --(attention / mean pooling)-->            genome embedding
             --(linear)-->                              one logit per antibiotic

Missing labels (genome not tested against an antibiotic) are masked out of the loss.
A Random Forest on genome-level k-mer frequencies is the classical baseline.

Run via:  python train.py --task amr [--test]
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from functools import partial

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from config import CONFIG
from dataset import _flatten_strings, build_protein_vocab, tokenize_protein_sequence
from model import ProteinBiLSTM, ProteinCausalLM, ProteinCNN


# ═══════════════════════════════════════════════════════════════════
# Labels
# ═══════════════════════════════════════════════════════════════════

def load_amr_labels(cfg: dict, genome_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return (labels, antibiotic_names). labels is float32 [n_genomes, n_antibiotics]
    with 1 = resistant, 0 = susceptible, -1 = not tested."""
    path = cfg["labels_file"]
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(CONFIG["dataset_name"], path, repo_type="dataset")
    df = pd.read_csv(path).drop_duplicates(cfg["genome_column"]).set_index(cfg["genome_column"])
    cols = list(cfg.get("antibiotics") or df.columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Unknown antibiotics {missing}; available: {list(df.columns)}")
    y = df.reindex(genome_names)[cols].to_numpy(dtype=np.float32)
    y = np.where(np.isnan(y), -1.0, y).astype(np.float32)
    bad = set(np.unique(y).tolist()) - {-1.0, 0.0, 1.0}
    if bad:
        raise ValueError(f"Labels must be 0/1/NaN, found {sorted(bad)}")
    return y, cols


# ═══════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════

class GenomeAMRDataset(Dataset):
    """One item = one genome: a sample of its proteins (tokenised) + its AMR label vector."""

    def __init__(
        self,
        hf_dataset,
        row_indices,
        labels: np.ndarray,
        *,
        protein_column: str,
        vocab,
        max_len: int,
        min_seq_len: int,
        max_proteins: int | None,
        train: bool,
        seed: int = 0,
    ):
        self.ds = hf_dataset
        self.rows = [int(r) for r in row_indices]
        self.labels = labels  # aligned with self.rows
        self.protein_column = protein_column
        self.vocab = vocab
        self.max_len = int(max_len)
        self.min_seq_len = int(min_seq_len)
        self.max_proteins = None if max_proteins is None else int(max_proteins)
        self.train = bool(train)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        row = self.ds[self.rows[i]]
        seqs = [
            s for s in _flatten_strings(row[self.protein_column])
            if isinstance(s, str) and len(s.strip()) >= max(self.min_seq_len, 1)
        ]
        if self.max_proteins is not None and len(seqs) > self.max_proteins:
            # Training: fresh sample every epoch. Validation: fixed sample per genome.
            rng = random.Random() if self.train else random.Random(self.seed + i)
            seqs = rng.sample(seqs, self.max_proteins)
        if not seqs:
            seqs = ["M"]
        ids = torch.stack(
            [tokenize_protein_sequence(s, vocab=self.vocab, max_len=self.max_len) for s in seqs]
        )
        return {"input_ids": ids, "labels": torch.from_numpy(self.labels[i])}


def collate_genomes(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    g = len(batch)
    p = max(b["input_ids"].shape[0] for b in batch)
    length = batch[0]["input_ids"].shape[1]
    ids = torch.full((g, p, length), pad_id, dtype=torch.long)
    protein_mask = torch.zeros((g, p), dtype=torch.bool)
    for k, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        ids[k, :n] = b["input_ids"]
        protein_mask[k, :n] = True
    return {
        "input_ids": ids,
        "protein_mask": protein_mask,
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class GenomeAMRClassifier(nn.Module):
    """Protein encoder (existing Transformer / BiLSTM / CNN) + genome pooling + AMR head."""

    def __init__(
        self,
        *,
        encoder_type: str,
        vocab_size: int,
        pad_id: int,
        max_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        num_antibiotics: int,
        pooling: str = "attn",
    ):
        super().__init__()
        if pooling not in ("attn", "mean"):
            raise ValueError(f"pooling must be 'attn' or 'mean', got {pooling}")
        self.pad_id = int(pad_id)
        self.pooling = pooling

        if encoder_type == "transformer":
            enc = ProteinCausalLM(
                vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                max_len=max_len, dropout=dropout, pad_id=pad_id,
            )
            dim = d_model
        elif encoder_type == "bilstm":
            enc = ProteinBiLSTM(
                vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, dropout=dropout, pad_id=pad_id
            )
            dim = d_model
        elif encoder_type == "cnn":
            enc = ProteinCNN(
                vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, dropout=dropout, pad_id=pad_id
            )
            dim = enc.combined_dim
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # The encoders switch to "classification mode" (pooled output) once a head is set;
        # replacing that head with Identity makes forward() return the pooled embedding.
        enc.set_classification_head(dim)
        enc.classifier = nn.Identity()
        self.encoder = enc

        self.attn = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim, num_antibiotics)

    def forward(self, input_ids: torch.Tensor, protein_mask: torch.Tensor) -> torch.Tensor:
        # input_ids: (G, P, L)   protein_mask: (G, P) True where a real protein sits
        flat = input_ids[protein_mask]                       # (N, L): only real proteins are encoded
        attn_mask = (flat != self.pad_id).long()
        emb = self.encoder(flat, attn_mask)                  # (N, D)
        out = emb.new_zeros(protein_mask.shape[0], protein_mask.shape[1], emb.shape[-1])
        out[protein_mask] = emb                              # (G, P, D)

        if self.pooling == "attn":
            scores = self.attn(out).squeeze(-1).masked_fill(~protein_mask, float("-inf"))
            w = torch.softmax(scores, dim=1).unsqueeze(-1)
            genome = (w * out).sum(dim=1)
        else:
            m = protein_mask.unsqueeze(-1).to(out.dtype)
            genome = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return self.head(self.drop(self.norm(genome)))       # (G, A) logits


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

def amr_metrics(scores: np.ndarray, y: np.ndarray, names: list[str], min_pos: int):
    """Per-antibiotic AUPRC / AUROC. scores are probabilities in [0,1]; y has -1 for missing."""
    rows = []
    for j, name in enumerate(names):
        m = y[:, j] >= 0
        t, s = y[m, j], scores[m, j]
        n_pos, n_neg = int((t == 1).sum()), int((t == 0).sum())
        r = {"antibiotic": name, "n": int(m.sum()), "n_resistant": n_pos, "auprc": None, "auroc": None}
        if n_pos > 0 and n_neg > 0:
            r["auprc"] = float(average_precision_score(t, s))
            r["auroc"] = float(roc_auc_score(t, s))
        rows.append(r)
    ok = [r for r in rows if r["auprc"] is not None and r["n_resistant"] >= min_pos]
    macro = {
        "auprc": float(np.mean([r["auprc"] for r in ok])) if ok else 0.0,
        "auroc": float(np.mean([r["auroc"] for r in ok])) if ok else 0.0,
        "n_evaluated": len(ok),
    }
    return rows, macro


# ═══════════════════════════════════════════════════════════════════
# Lightning module
# ═══════════════════════════════════════════════════════════════════

class LitAMR(pl.LightningModule):
    def __init__(
        self,
        *,
        encoder_type: str,
        vocab_size: int,
        pad_id: int,
        max_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        lr: float,
        weight_decay: float,
        antibiotics: list[str],
        pos_weight: torch.Tensor | None = None,
        pooling: str = "attn",
        min_positives: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])
        self.antibiotics = list(antibiotics)
        self.min_positives = int(min_positives)
        self.model = GenomeAMRClassifier(
            encoder_type=encoder_type, vocab_size=vocab_size, pad_id=pad_id, max_len=max_len,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
            num_antibiotics=len(self.antibiotics), pooling=pooling,
        )
        pw = pos_weight if pos_weight is not None else torch.ones(len(self.antibiotics))
        self.register_buffer("pos_weight", pw.float())
        self._val_scores: list[torch.Tensor] = []
        self._val_y: list[torch.Tensor] = []
        self.metrics_path: str | None = None
        self.last_rows: list[dict] = []
        self.last_macro: dict = {}

    def forward(self, input_ids, protein_mask):
        return self.model(input_ids, protein_mask)

    def _masked_bce(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mask = (y >= 0).float()
        loss = F.binary_cross_entropy_with_logits(
            logits, y.clamp(min=0), pos_weight=self.pos_weight, reduction="none"
        )
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["protein_mask"])
        loss = self._masked_bce(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=logits.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["protein_mask"])
        loss = self._masked_bce(logits, batch["labels"])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=logits.shape[0])
        self._val_scores.append(torch.sigmoid(logits).detach().float().cpu())
        self._val_y.append(batch["labels"].detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        if not self._val_scores:
            return
        scores = torch.cat(self._val_scores).numpy()
        y = torch.cat(self._val_y).numpy()
        self._val_scores.clear()
        self._val_y.clear()
        rows, macro = amr_metrics(scores, y, self.antibiotics, self.min_positives)
        self.last_rows, self.last_macro = rows, macro
        self.log("val_auprc", macro["auprc"], prog_bar=True)
        self.log("val_auroc", macro["auroc"], prog_bar=True)
        if self.metrics_path and not self.trainer.sanity_checking:
            with open(self.metrics_path, "w") as f:
                json.dump({"macro": macro, "per_antibiotic": rows, "epoch": int(self.current_epoch)}, f, indent=2)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=float(self.hparams.lr), weight_decay=float(self.hparams.weight_decay)
        )


# ═══════════════════════════════════════════════════════════════════
# Random Forest baseline (genome-level k-mer frequencies)
# ═══════════════════════════════════════════════════════════════════

_AA = "ACDEFGHIKLMNPQRSTVWY"
_LUT = np.full(256, -1, dtype=np.int16)
for _i, _c in enumerate(_AA):
    _LUT[ord(_c)] = _i


def genome_kmer_features(seqs: list[str]) -> np.ndarray:
    """1/2/3-mer frequencies pooled over a genome's proteins + log mean length (8421 dims)."""
    c1, c2, c3 = np.zeros(20), np.zeros(400), np.zeros(8000)
    total = 0
    for s in seqs:
        a = _LUT[np.frombuffer(s.strip().upper().encode("ascii", "ignore"), dtype=np.uint8)]
        a = a[a >= 0].astype(np.int64)
        if len(a) == 0:
            continue
        c1 += np.bincount(a, minlength=20)
        if len(a) >= 2:
            c2 += np.bincount(a[:-1] * 20 + a[1:], minlength=400)
        if len(a) >= 3:
            c3 += np.bincount((a[:-2] * 20 + a[1:-1]) * 20 + a[2:], minlength=8000)
        total += len(a)
    f = [c1 / max(c1.sum(), 1), c2 / max(c2.sum(), 1), c3 / max(c3.sum(), 1)]
    f.append(np.array([math.log1p(total / max(len(seqs), 1))]))
    return np.concatenate(f).astype(np.float32)


def run_random_forest(ds, rows, y, names, train_pos, val_pos, *, cfg, seed, out_dir, n_estimators, min_pos):
    from sklearn.ensemble import RandomForestClassifier

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731

    print("\n" + "=" * 20 + "\nRandom Forest (genome k-mer features)\n" + "=" * 20)
    max_p = cfg.get("max_proteins_per_genome")
    X = np.zeros((len(rows), 20 + 400 + 8000 + 1), dtype=np.float32)
    for k, r in enumerate(tqdm(rows, desc="k-mer features")):
        seqs = [s for s in _flatten_strings(ds[int(r)][CONFIG["protein_column"]]) if isinstance(s, str)]
        if max_p is not None and len(seqs) > max_p:
            seqs = random.Random(seed + k).sample(seqs, max_p)
        X[k] = genome_kmer_features(seqs)

    per = []
    for j, name in enumerate(names):
        tr = np.array([p for p in train_pos if y[p, j] >= 0])
        va = np.array([p for p in val_pos if y[p, j] >= 0])
        r = {"antibiotic": name, "n": int(len(va)), "n_resistant": int((y[va, j] == 1).sum()) if len(va) else 0,
             "auprc": None, "auroc": None}
        if len(tr) and len(va) and len(set(y[tr, j])) == 2 and len(set(y[va, j])) == 2:
            rf = RandomForestClassifier(
                n_estimators=n_estimators, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
            )
            rf.fit(X[tr], y[tr, j])
            s = rf.predict_proba(X[va])[:, 1]
            r["auprc"] = float(average_precision_score(y[va, j], s))
            r["auroc"] = float(roc_auc_score(y[va, j], s))
            print(f"  {name:32s} AUPRC {r['auprc']:.3f}  AUROC {r['auroc']:.3f}  (val n={r['n']}, R={r['n_resistant']})")
        per.append(r)
    ok = [r for r in per if r["auprc"] is not None and r["n_resistant"] >= min_pos]
    macro = {
        "auprc": float(np.mean([r["auprc"] for r in ok])) if ok else 0.0,
        "auroc": float(np.mean([r["auroc"] for r in ok])) if ok else 0.0,
        "n_evaluated": len(ok),
    }
    print(f"RF macro AUPRC {macro['auprc']:.3f}  AUROC {macro['auroc']:.3f}  over {macro['n_evaluated']} antibiotics")
    with open(os.path.join(out_dir, "amr_metrics_random_forest.json"), "w") as f:
        json.dump({"macro": macro, "per_antibiotic": per}, f, indent=2)
    return macro


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

def run_amr(test: bool = False, models: list[str] | None = None, overrides: dict | None = None) -> None:
    import datasets
    from pytorch_lightning.callbacks import ModelCheckpoint

    cfg = dict(CONFIG["amr"])
    models = models or ["transformer", "bilstm", "cnn", "random_forest"]
    num_workers = int(CONFIG["num_workers"])
    limit_train, limit_val = 1.0, 1.0
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    if "limit_batches" in overrides:                  # timing / memory probes
        limit_train = limit_val = int(overrides.pop("limit_batches"))
    if "num_workers" in overrides:
        num_workers = int(overrides.pop("num_workers"))
    if test:
        print("\n!!! AMR TEST MODE !!!")
        cfg.update(epochs=1, max_proteins_per_genome=8, rf_estimators=20, checkpoint_dir=os.path.join(cfg["checkpoint_dir"], "test_run"))
        limit_train, limit_val = 2, 2
        num_workers = 0
    cfg.update(overrides)                             # explicit CLI values win over the test preset
    if overrides:
        print(f"AMR config overrides: {overrides}")
    out_dir = cfg["checkpoint_dir"]
    os.makedirs(out_dir, exist_ok=True)

    os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(".cache", "huggingface")))
    os.environ.setdefault("HF_HUB_CACHE", os.path.abspath(os.path.join(".cache", "huggingface", "hub")))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath(os.path.join(".cache", "huggingface", "datasets")))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    seed = int(CONFIG["seed"])
    pl.seed_everything(seed, workers=True)
    vocab = build_protein_vocab()

    ds = datasets.load_dataset(CONFIG["dataset_name"], split=CONFIG["dataset_split"])
    target = str(CONFIG["species_filter"])
    keep = [i for i, t in enumerate(ds[CONFIG["species_column"]]) if str(t) == target]
    ds = ds.select(keep)
    print(f"Filtered to taxid {target}: {len(ds)} genomes")

    genome_names = list(ds[cfg["genome_column"]])
    y_all, antibiotics = load_amr_labels(cfg, genome_names)
    rows = np.where((y_all >= 0).any(axis=1))[0]
    print(f"{len(rows)} genomes have at least one AMR label; {len(antibiotics)} antibiotics")
    if test:
        rows = rows[:40]

    perm = np.random.RandomState(seed).permutation(len(rows))
    n_val = max(1, int(len(rows) * float(CONFIG.get("val_fraction", 0.1))))
    val_pos, train_pos = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    y = y_all[rows]                                   # aligned with `rows`
    print(f"Genome split: train {len(train_pos)}  val {len(val_pos)}")
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({"train": [genome_names[rows[p]] for p in train_pos],
                   "val": [genome_names[rows[p]] for p in val_pos], "antibiotics": antibiotics}, f)

    ytr = y[train_pos]
    pos, neg = (ytr == 1).sum(0), (ytr == 0).sum(0)
    pos_weight = torch.tensor(
        np.clip(np.where(pos > 0, neg / np.maximum(pos, 1), 1.0), 1.0, float(cfg["max_pos_weight"])),
        dtype=torch.float32,
    )

    def make_ds(positions, train):
        return GenomeAMRDataset(
            ds, rows[positions], y[positions],
            protein_column=CONFIG["protein_column"], vocab=vocab, max_len=CONFIG["max_len"],
            min_seq_len=CONFIG.get("min_seq_len", 0), max_proteins=cfg["max_proteins_per_genome"],
            train=train, seed=seed + (0 if train else 999_983),
        )

    collate = partial(collate_genomes, pad_id=vocab.pad_id)

    def make_dl(dataset, shuffle):
        kw = {"prefetch_factor": 2, "persistent_workers": True} if num_workers > 0 else {}
        return DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True, collate_fn=collate, **kw)

    summary: dict[str, dict] = {}
    for model_type in [m for m in models if m != "random_forest"]:
        print(f"\n{'=' * 20}\nTraining AMR model: {model_type}\n{'=' * 20}")
        mdir = os.path.join(out_dir, model_type)
        os.makedirs(mdir, exist_ok=True)
        lit = LitAMR(
            encoder_type=model_type, vocab_size=vocab.size, pad_id=vocab.pad_id, max_len=CONFIG["max_len"],
            d_model=CONFIG["d_model"], n_heads=CONFIG["n_heads"], n_layers=CONFIG["n_layers"],
            dropout=CONFIG["dropout"], lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
            antibiotics=antibiotics, pos_weight=pos_weight, pooling=cfg["pooling"],
            min_positives=int(cfg["min_positives_eval"]),
        )
        lit.metrics_path = os.path.join(mdir, "val_metrics_latest.json")
        ckpt = ModelCheckpoint(
            dirpath=mdir, filename="epoch_{epoch:04d}_auprc_{val_auprc:.4f}", monitor="val_auprc",
            mode="max", save_last=True, save_top_k=1, auto_insert_metric_name=False,
        )
        trainer = pl.Trainer(
            max_epochs=int(cfg["epochs"]), accelerator="auto", devices="auto",
            precision=CONFIG["precision"], callbacks=[ckpt], log_every_n_steps=10,
            accumulate_grad_batches=int(cfg["accumulate_grad_batches"]),
            limit_train_batches=limit_train, limit_val_batches=limit_val,
            default_root_dir=out_dir,
        )
        val_dl = make_dl(make_ds(val_pos, False), False)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        trainer.fit(lit, make_dl(make_ds(train_pos, True), True), val_dl)
        fit_s = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        print(f"[{model_type}] fit took {fit_s:.0f}s ({fit_s / max(trainer.current_epoch, 1):.0f}s/epoch), "
              f"peak GPU memory {peak:.1f} GiB, {trainer.global_step} optimizer steps")

        # Final evaluation with the best checkpoint; this writes the per-antibiotic report.
        lit.metrics_path = os.path.join(out_dir, f"amr_metrics_{model_type}.json")
        trainer.validate(lit, val_dl, ckpt_path="best")
        summary[model_type] = lit.last_macro
        del lit, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "random_forest" in models:
        try:
            summary["random_forest"] = run_random_forest(
                ds, rows, y, antibiotics, train_pos, val_pos, cfg=cfg, seed=seed, out_dir=out_dir,
                n_estimators=int(cfg["rf_estimators"]), min_pos=int(cfg["min_positives_eval"]),
            )
        except Exception as e:  # keep the deep-model results even if RF fails
            import traceback

            print(f"Random Forest failed: {e}")
            traceback.print_exc()

    with open(os.path.join(out_dir, "amr_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nAMR summary (macro over antibiotics with enough resistant genomes):")
    for k, v in summary.items():
        print(f"  {k:14s} AUPRC {v.get('auprc', 0):.3f}  AUROC {v.get('auroc', 0):.3f}  (n={v.get('n_evaluated', 0)})")
