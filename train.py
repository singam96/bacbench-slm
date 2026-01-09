import os

import datasets
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from config import CONFIG
from data_module import ProteinDataModule
from dataset import build_protein_vocab
from lit_module import LitProteinCausalLM


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

if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(".cache", "huggingface")))
    os.environ.setdefault("HF_HUB_CACHE", os.path.abspath(os.path.join(".cache", "huggingface", "hub")))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath(os.path.join(".cache", "huggingface", "datasets")))

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    pl.seed_everything(int(CONFIG["seed"]), workers=True)
    vocab = build_protein_vocab()

    ds = datasets.load_dataset(CONFIG["dataset_name"], split=CONFIG["dataset_split"])
    splits = _kfold_splits(len(ds), int(CONFIG["k_folds"]), seed=int(CONFIG["seed"]))

    fold_cfg = int(CONFIG["fold"])
    fold_ids = range(len(splits)) if fold_cfg < 0 else [fold_cfg]

    for fold in fold_ids:
        train_genomes, val_genomes = splits[int(fold)]

        fold_dir = os.path.join(CONFIG["checkpoint_dir"], f"fold_{int(fold)}")
        os.makedirs(fold_dir, exist_ok=True)

        dm = ProteinDataModule(
            hf_dataset=ds,
            train_genome_indices=train_genomes,
            val_genome_indices=val_genomes,
            protein_column=CONFIG["protein_column"],
            vocab=vocab,
            max_len=CONFIG["max_seq_len"],
            min_seq_len=CONFIG["min_seq_len"],
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
        )

        model = LitProteinCausalLM(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_len=CONFIG["max_seq_len"],
            d_model=CONFIG["d_model"],
            n_heads=CONFIG["n_heads"],
            n_layers=CONFIG["n_layers"],
            dropout=CONFIG["dropout"],
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
        )

        ckpt_callback = ModelCheckpoint(
            dirpath=fold_dir,
            filename="epoch_{epoch:04d}_val_{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_last=True,
            save_top_k=1,
        )

        trainer = pl.Trainer(
            max_epochs=CONFIG["epochs"],
            accelerator="auto",
            devices="auto",
            precision=CONFIG["precision"],
            callbacks=[ckpt_callback],
            log_every_n_steps=10,
        )

        trainer.fit(model, datamodule=dm)
