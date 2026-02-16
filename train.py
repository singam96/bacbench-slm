import os

import datasets
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

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
    models_to_train = ["transformer", "bilstm"] # We can add more here or make it configurable

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

    # Generate Report (if available)
    if generate_report:
        try:
            generate_report()
        except Exception as e:
            print(f"Error generating report: {e}")
            print("Please ensure pandas, matplotlib are installed: pip install pandas matplotlib")
    else:
        print("Skipping report generation (create_report.py not found or dependencies missing)")
