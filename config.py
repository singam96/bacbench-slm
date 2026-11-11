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
    "vocab_size": 24,  # 20 AA + 4 special tokens (train.py uses vocab.size)
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

    # Genome-level AMR prediction (python train.py --task amr). See amr.py.
    # One model predicts every antibiotic in the label table at once; missing
    # labels are masked out of the loss.
    "amr": {
        "labels_file": "binary_labels.csv",   # in the dataset repo; 1 = resistant, 0 = susceptible
        "genome_column": "genome_name",
        "antibiotics": None,                  # None = all columns, or a list of names
        "max_proteins_per_genome": 64,        # proteins sampled per genome (resampled each epoch in training)
        "batch_size": 2,                      # GENOMES per step (x proteins x max_len tokens each)
        "accumulate_grad_batches": 4,
        "epochs": 10,
        "pooling": "attn",                    # 'attn' or 'mean' over protein embeddings
        "min_positives_eval": 10,             # antibiotics with fewer resistant genomes in val are shown but not averaged
        "max_pos_weight": 10.0,
        "rf_estimators": 200,
        "checkpoint_dir": "checkpoints_amr",
    },
}
