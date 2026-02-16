import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import GenomeProteinIndexDataset, ProteinVocab


class ProteinDataModule(pl.LightningDataModule):
    def __init__(
        self,
        hf_dataset,
        train_genome_indices: list[int],
        val_genome_indices: list[int],
        protein_column: str,
        vocab: ProteinVocab,
        max_len: int,
        min_seq_len: int,
        max_proteins_per_genome: int | None,
        batch_size: int,
        num_workers: int = 0,
        seed: int = 0,
        label_column: str | None = None,
        label_map: dict | None = None,
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.train_genome_indices = list(train_genome_indices)
        self.val_genome_indices = list(val_genome_indices)
        self.protein_column = str(protein_column)
        self.vocab = vocab
        self.max_len = int(max_len)
        self.min_seq_len = int(min_seq_len)
        self.max_proteins_per_genome = None if max_proteins_per_genome is None else int(max_proteins_per_genome)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = int(seed)
        self.label_column = label_column
        self.label_map = label_map

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        self.train_dataset = GenomeProteinIndexDataset(
            self.hf_dataset,
            self.train_genome_indices,
            protein_column=self.protein_column,
            vocab=self.vocab,
            max_len=self.max_len,
            min_seq_len=self.min_seq_len,
            max_proteins_per_genome=self.max_proteins_per_genome,
            seed=self.seed,
            label_column=self.label_column,
            label_map=self.label_map,
        )
        self.val_dataset = GenomeProteinIndexDataset(
            self.hf_dataset,
            self.val_genome_indices,
            protein_column=self.protein_column,
            vocab=self.vocab,
            max_len=self.max_len,
            min_seq_len=self.min_seq_len,
            max_proteins_per_genome=self.max_proteins_per_genome,
            seed=self.seed + 999_983,
            label_column=self.label_column,
            label_map=self.label_map,
        )

    def train_dataloader(self):
        dl_kwargs = {}
        if self.num_workers > 0:
            dl_kwargs["prefetch_factor"] = 2
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            **dl_kwargs,
        )

    def val_dataloader(self):
        dl_kwargs = {}
        if self.num_workers > 0:
            dl_kwargs["prefetch_factor"] = 2
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            **dl_kwargs,
        )
