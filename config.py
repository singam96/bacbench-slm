import torch

CONFIG = {
    # Dataset
    "dataset_name": "macwiatrak/bacbench-antibiotic-resistance-protein-sequences",
    "dataset_split": "train",
    "protein_column": "protein_sequence",
    "species_column": "taxid",
    "species_filter": "562",
    "max_len": 512,
    "batch_size": 32,
    "epochs": 3,  # Increased epochs since dataset is smaller
    "max_steps": 50_000,
    "limit_train_batches": 1.0,
    "limit_val_batches": 1.0,
    "max_proteins_per_genome": 64,
    "lr": 3e-4,
    "weight_decay": 0.01,
    
    # Classification settings
    "task_type": "classification", # 'classification' or 'causal_lm'
    "label_column": "product",
    "num_classes": 15, # Will be set dynamically if possible, or we pick top-N common classes
    "top_n_classes": 15, # Use top N most frequent products, map rest to <unk>

    # Model
    "vocab_size": 26,  # 20 AA + special tokens
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "dropout": 0.1,
    "seed": 42,
    
    # Training
    "checkpoint_dir": "checkpoints",
    "num_workers": 4,
    "precision": "32", # Use full precision to avoid NaNs
    "accumulate_grad_batches": 1,
    "k_folds": 1,
    "val_fraction": 0.1,  # Fraction of data to use for validation if k_folds <= 1
}
