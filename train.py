import json
import math
import os

import datasets
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import CONFIG
from data_module import ProteinDataModule
from dataset import build_protein_vocab, _flatten_strings
from lit_module import LitProteinCausalLM

# Optional: Import additional training scripts
try:
    from create_report import generate_report
except ImportError:
    generate_report = None


def _simple_split(n: int, val_frac: float, *, seed: int) -> list[tuple[list[int], list[int]]]:
    if not (0 < val_frac < 1):
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    idxs = list(range(int(n)))
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(idxs), generator=g).tolist()
    idxs = [idxs[i] for i in perm]
    
    n_val = int(n * val_frac)
    val = idxs[:n_val]
    train = idxs[n_val:]
    return [(train, val)]


def _kfold_splits(n: int, k: int, *, seed: int) -> list[tuple[list[int], list[int]]]:
    if k <= 1:
        raise ValueError(f"k must be > 1, got {k}")
    idxs = list(range(int(n)))
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(idxs), generator=g).tolist()
    idxs = [idxs[i] for i in perm]

    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1

    splits: list[tuple[list[int], list[int]]] = []
    start = 0
    for fs in fold_sizes:
        val = idxs[start : start + fs]
        train = idxs[:start] + idxs[start + fs :]
        splits.append((train, val))
        start += fs
    return splits

import argparse


# ═══════════════════════════════════════════════════════════════════
# Random Forest baseline
# ═══════════════════════════════════════════════════════════════════

AMINO_ACIDS_ORDERED = "ACDEFGHIKLMNPQRSTVWY"


def _kmer_features(seq: str, k: int) -> dict[str, float]:
    """Count normalised k-mer frequencies for a single protein sequence."""
    seq = seq.strip().upper()
    if len(seq) < k:
        return {}
    counts: dict[str, float] = {}
    for i in range(len(seq) - k + 1):
        mer = seq[i : i + k]
        counts[mer] = counts.get(mer, 0.0) + 1.0
    n_kmers = len(seq) - k + 1
    for mer in counts:
        counts[mer] /= n_kmers
    return counts


def _extract_protein_features(seq: str) -> dict[str, float]:
    """Build a feature dict from a single protein sequence."""
    feat: dict[str, float] = {}
    # 1-mer (amino acid composition)
    feat.update(_kmer_features(seq, k=1))
    # 2-mer (dipeptide composition)
    feat.update(_kmer_features(seq, k=2))
    # 3-mer (tripeptide composition)
    feat.update(_kmer_features(seq, k=3))
    # Sequence length (log-scaled so RF can use it)
    feat["seq_len"] = math.log(max(len(seq.strip()), 1))
    return feat


def _build_rf_dataset(
    hf_dataset,
    genome_indices: list[int],
    *,
    protein_column: str,
    label_column: str,
    label_map: dict[str, int],
    max_proteins_per_genome: int | None = None,
    min_seq_len: int = 0,
    seed: int = 0,
) -> tuple[list[dict[str, float]], list[int]]:
    """Iterate genomes and extract features + labels for every protein."""
    features: list[dict[str, float]] = []
    labels: list[int] = []
    rng = torch.Generator().manual_seed(int(seed))

    for row_pos, gi in enumerate(genome_indices):
        row = hf_dataset[int(gi)]
        seqs = list(_flatten_strings(row.get(protein_column)))
        prot_labels = list(_flatten_strings(row.get(label_column)))

        # Build index list for valid sequences
        valid = [i for i, s in enumerate(seqs) if isinstance(s, str) and len(s.strip()) >= min_seq_len]
        if max_proteins_per_genome is not None and len(valid) > max_proteins_per_genome:
            perm = torch.randperm(len(valid), generator=rng).tolist()
            valid = [valid[i] for i in perm[:max_proteins_per_genome]]

        for idx in valid:
            seq = str(seqs[idx]).strip()
            feat = _extract_protein_features(seq)

            # Label
            raw_label = str(prot_labels[idx]).strip() if idx < len(prot_labels) else ""
            label_id = label_map.get(raw_label, -100)
            if label_id == -100:
                continue  # skip unknown / ignored labels

            features.append(feat)
            labels.append(label_id)

    return features, labels


def train_random_forest(
    hf_dataset,
    train_genome_indices: list[int],
    val_genome_indices: list[int],
    *,
    protein_column: str,
    label_column: str,
    label_map: dict[str, int],
    max_proteins_per_genome: int | None = None,
    min_seq_len: int = 0,
    seed: int = 0,
    model_dir: str = "checkpoints",
    n_estimators: int = 300,
    max_depth: int | None = None,
    n_jobs: int = -1,
) -> dict:
    """Train a Random Forest classifier on k-mer features and evaluate."""
    print("\n" + "=" * 20)
    print("Training Random Forest baseline")
    print("=" * 20)

    print("Extracting training features...")
    train_feat_dicts, train_labels = _build_rf_dataset(
        hf_dataset,
        train_genome_indices,
        protein_column=protein_column,
        label_column=label_column,
        label_map=label_map,
        max_proteins_per_genome=max_proteins_per_genome,
        min_seq_len=min_seq_len,
        seed=seed,
    )

    print("Extracting validation features...")
    val_feat_dicts, val_labels = _build_rf_dataset(
        hf_dataset,
        val_genome_indices,
        protein_column=protein_column,
        label_column=label_column,
        label_map=label_map,
        max_proteins_per_genome=max_proteins_per_genome,
        min_seq_len=min_seq_len,
        seed=seed + 999_983,
    )

    print(f"Train samples: {len(train_labels)}   Val samples: {len(val_labels)}")
    print(f"Number of classes: {len(set(train_labels))}")

    # Vectorise features (DictVectorizer handles sparse output)
    vec = DictVectorizer(sparse=True)
    X_train = vec.fit_transform(train_feat_dicts)
    X_val = vec.transform(val_feat_dicts)
    print(f"Feature dimension: {X_train.shape[1]}")

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    # Train
    print(f"Training RF ({n_estimators} trees, max_depth={max_depth})...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=int(seed),
        class_weight="balanced",
        verbose=0,
    )
    rf.fit(X_train, y_train)

    # Evaluate
    train_preds = rf.predict(X_train)
    val_preds = rf.predict(X_val)

    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="weighted")

    print(f"RF Train Accuracy: {train_acc:.4f}")
    print(f"RF Val Accuracy:   {val_acc:.4f}")
    print(f"RF Val F1 (weighted): {val_f1:.4f}")

    # Save results for the report
    result = {
        "Model": "random_forest",
        "Type": "Feature-based (k-mer)",
        "Fold": "all",
        "Val Accuracy": round(val_acc, 4),
        "Val F1 (weighted)": round(val_f1, 4),
    }
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "rf_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("RF metrics saved to checkpoints/rf_metrics.json")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode (small dataset, few epochs)")
    args = parser.parse_args()

    if args.test:
        print("\n!!! TEST MODE ACTIVATED !!!")
        print("Overriding configuration for smoke test...")
        CONFIG["epochs"] = 1
        CONFIG["max_steps"] = 10
        CONFIG["limit_train_batches"] = 2
        CONFIG["limit_val_batches"] = 2
        CONFIG["checkpoint_dir"] = os.path.join(CONFIG.get("checkpoint_dir", "checkpoints"), "test_run")
        # Ensure we don't try to use k-fold in test mode if it complicates things
        CONFIG["k_folds"] = 1
        # Use more classes in test mode to reduce ignored targets
        CONFIG["top_n_classes"] = 1000
    
    os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(".cache", "huggingface")))
    os.environ.setdefault("HF_HUB_CACHE", os.path.abspath(os.path.join(".cache", "huggingface", "hub")))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath(os.path.join(".cache", "huggingface", "datasets")))

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    pl.seed_everything(int(CONFIG["seed"]), workers=True)
    vocab = build_protein_vocab()

    ds = datasets.load_dataset(CONFIG["dataset_name"], split=CONFIG["dataset_split"])

    if CONFIG.get("species_filter"):
        species_col = CONFIG.get("species_column", "taxid")
        target = CONFIG["species_filter"]
        print(f"Filtering dataset for species ID: '{target}' in column '{species_col}'")
        
        # Optimized filtering (we know taxid is often int, so cast to str)
        # If test mode, just take first 100 genomes to speed up
        if args.test:
            print("Test mode: Taking first 50 genomes BEFORE filtering (may yield empty if species not in top 50)")
            # Wait, if we take top 50 and they aren't E. coli, we get empty.
            # Better to filter first then slice, OR use streaming with take?
            # Since dataset is cached, filtering is fast.
            # Let's just slice AFTER filtering.
            pass

        ds = ds.filter(
            lambda x: str(x[species_col]) == target,
            num_proc=4
        )
        print(f"Filtered dataset: {len(ds)} genomes")
        
        if args.test:
             print("Test mode: Slicing dataset to 20 genomes.")
             ds = ds.select(range(min(len(ds), 20)))
    
    # Classification Label Setup
    label_map = None
    if CONFIG.get("task_type") == "classification":
        print("Setting up classification labels...")
        label_col = CONFIG["label_column"]
        top_n = CONFIG.get("top_n_classes", 15)
        
        from collections import Counter
        from tqdm import tqdm
        
        # In test mode, we might not see all classes if we filter first.
        # But we need consistent labels.
        # Let's scan the filtered DS.
        product_counter = Counter()
        
        # For label mapping, we should ideally use the WHOLE dataset or a fixed list
        # to ensure consistency. But for now let's scan what we have.
        print("Scanning dataset for labels...")
        # Use a small subset for scanning if in test mode to be fast, 
        # BUT this might miss classes if we are extremely unlucky.
        # Actually, if we want to test the PIPELINE, we need valid labels.
        scan_ds = ds
        
        for row in tqdm(scan_ds, desc="Scanning labels"):
            p_list = row[label_col]
            for p in _flatten_strings(p_list):
                 if p:
                     product_counter[p.strip()] += 1
        
        most_common = product_counter.most_common(top_n)
        print("Top classes found:", most_common)
        
        label_map = {label: i for i, (label, _) in enumerate(most_common)}
        CONFIG["num_classes"] = len(label_map)
        print(f"Classification configured with {len(label_map)} classes.")

    k_folds = int(CONFIG.get("k_folds", 1))
    if k_folds > 1:
        print(f"Using {k_folds}-fold cross-validation")
        splits = _kfold_splits(len(ds), k_folds, seed=int(CONFIG["seed"]))
    else:
        val_frac = float(CONFIG.get("val_fraction", 0.1))
        print(f"Using simple train/val split (val_fraction={val_frac})")
        splits = _simple_split(len(ds), val_frac, seed=int(CONFIG["seed"]))

    fold_cfg = int(CONFIG.get("fold", 0))
    fold_ids = range(len(splits)) if fold_cfg < 0 else [fold_cfg]

    # Model Loop
    models_to_train = ["transformer", "bilstm", "cnn"] # We can add more here or make it configurable

    for model_type in models_to_train:
        print(f"\n{'='*20}\nTraining Model: {model_type}\n{'='*20}")
        
        for fold in fold_ids:
            train_genomes, val_genomes = splits[int(fold)]
    
            fold_dir = os.path.join(CONFIG["checkpoint_dir"], f"{model_type}_fold_{int(fold)}")
            os.makedirs(fold_dir, exist_ok=True)
    
            dm = ProteinDataModule(
                hf_dataset=ds,
                train_genome_indices=train_genomes,
                val_genome_indices=val_genomes,
                protein_column=CONFIG["protein_column"],
                vocab=vocab,
                max_len=CONFIG["max_len"],
                min_seq_len=CONFIG.get("min_seq_len", 0),
                max_proteins_per_genome=CONFIG.get("max_proteins_per_genome"),
                batch_size=CONFIG["batch_size"],
                num_workers=CONFIG["num_workers"],
                seed=CONFIG["seed"],
                label_column=CONFIG.get("label_column"),
                label_map=label_map,
            )
    
            model = LitProteinCausalLM(
                vocab_size=vocab.size,
                pad_id=vocab.pad_id,
                max_len=CONFIG["max_len"],
                d_model=CONFIG["d_model"],
                n_heads=CONFIG["n_heads"],
                n_layers=CONFIG["n_layers"],
                dropout=CONFIG["dropout"],
                lr=CONFIG["lr"],
                weight_decay=CONFIG["weight_decay"],
                model_type=model_type,
                task_type=CONFIG.get("task_type", "causal_lm"),
                num_classes=CONFIG.get("num_classes"),
            )
    
            ckpt_callback = ModelCheckpoint(
                dirpath=fold_dir,
                filename="epoch_{epoch:04d}_loss_{val_loss:.4f}_acc_{val_acc:.4f}",
                monitor="val_loss",
                mode="min",
                save_last=True,
                save_top_k=1,
                auto_insert_metric_name=False, # Prevent "epoch=0000" prefix
            )
    
            trainer = pl.Trainer(
                max_epochs=CONFIG["epochs"],
                max_steps=int(CONFIG["max_steps"]) if int(CONFIG.get("max_steps", -1)) > 0 else -1,
                accelerator="auto",
                devices="auto",
                precision=CONFIG["precision"],
                callbacks=[ckpt_callback],
                log_every_n_steps=10,
                limit_train_batches=CONFIG.get("limit_train_batches", 1.0),
                limit_val_batches=CONFIG.get("limit_val_batches", 1.0),
            )
    
            trainer.fit(model, datamodule=dm)

    # ── Random Forest baseline ──────────────────────────────────
    if CONFIG.get("task_type") == "classification" and label_map is not None:
        # Use the first fold's split (or the only split)
        train_genomes, val_genomes = splits[0]
        try:
            rf_result = train_random_forest(
                hf_dataset=ds,
                train_genome_indices=train_genomes,
                val_genome_indices=val_genomes,
                protein_column=CONFIG["protein_column"],
                label_column=CONFIG["label_column"],
                label_map=label_map,
                max_proteins_per_genome=CONFIG.get("max_proteins_per_genome"),
                min_seq_len=CONFIG.get("min_seq_len", 0),
                seed=CONFIG["seed"],
                model_dir=CONFIG["checkpoint_dir"],
                n_estimators=100 if args.test else 300,
                max_depth=10 if args.test else None,
            )
        except Exception as e:
            print(f"Error training Random Forest: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping Random Forest baseline (not a classification task or no labels)")

    # Generate Report (if available)
    if generate_report:
        try:
            generate_report()
        except Exception as e:
            print(f"Error generating report: {e}")
            print("Please ensure pandas, matplotlib are installed: pip install pandas matplotlib")
    else:
        print("Skipping report generation (create_report.py not found or dependencies missing)")
