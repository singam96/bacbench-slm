import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import glob
from config import CONFIG

def parse_ckpt_metrics(filename: str) -> dict | None:
    """Parse epoch/loss/acc from a Lightning checkpoint filename.

    Expected format: epoch_0002_loss_0.3676_acc_0.8544.ckpt
    Returns None for files like last.ckpt / last-v1.ckpt that carry no metrics.
    """
    if not filename.endswith(".ckpt"):
        return None
    name = filename[: -len(".ckpt")]
    if name.startswith("last"):
        return None  # last.ckpt / last-v1.ckpt have no metrics in the name
    parts = name.split("_")
    metrics: dict = {}
    for i, part in enumerate(parts):
        if part == "epoch" and i + 1 < len(parts):
            try:
                metrics["epoch"] = int(parts[i + 1])
            except ValueError:
                pass
        elif part == "loss" and i + 1 < len(parts):
            try:
                metrics["val_loss"] = float(parts[i + 1])
            except ValueError:
                pass
        elif part == "acc" and i + 1 < len(parts):
            try:
                metrics["val_acc"] = float(parts[i + 1])
            except ValueError:
                pass
    if "val_loss" not in metrics and "val_acc" not in metrics:
        return None
    return metrics


def best_checkpoint_metrics(fold_dir: str) -> dict | None:
    """Return metrics of the best (lowest val_loss) epoch checkpoint in a folder."""
    best = None
    for ckpt in glob.glob(os.path.join(fold_dir, "*.ckpt")):
        m = parse_ckpt_metrics(os.path.basename(ckpt))
        if m is None:
            continue
        if best is None or m.get("val_loss", float("inf")) < best.get("val_loss", float("inf")):
            best = m
    return best


def generate_report():
    print(f"\n{'='*20}\nGenerating Model Comparison Report\n{'='*20}")
    
    report_data = []
    
    # 1. Gather Deep Learning Model Metrics (Transformer, BiLSTM, CNN)
    # Metrics are parsed from the epoch checkpoint filenames. The `last.ckpt`
    # files are excluded because they carry no metrics in their names.
    checkpoint_dir = CONFIG["checkpoint_dir"]
    
    # DL Models (Transformer, BiLSTM, CNN)
    for model_type in ["transformer", "bilstm", "cnn"]:
        fold_dirs = glob.glob(os.path.join(checkpoint_dir, f"{model_type}_fold_*"))
        
        for fd in fold_dirs:
            m = best_checkpoint_metrics(fd)
            if m is None:
                print(f"  [warn] No metric-bearing checkpoint found in {fd}")
                continue
            report_data.append({
                "Model": model_type,
                "Type": "Deep Learning",
                "Fold": os.path.basename(fd),
                "Val Loss": m.get("val_loss"),
                "Val Accuracy": m.get("val_acc", "N/A"),
                "Convergence": "See charts below",
            })

    # 2. Load Random Forest metrics (saved by train.py)
    rf_metrics_path = os.path.join(checkpoint_dir, "rf_metrics.json")
    rf_data: dict = {}
    if os.path.exists(rf_metrics_path):
        with open(rf_metrics_path) as f:
            rf_data = json.load(f)
        report_data.append({
            "Model": rf_data.get("Model", "random_forest"),
            "Type": rf_data.get("Type", "Feature-based (k-mer)"),
            "Fold": rf_data.get("Fold", "all"),
            "Val Loss": "N/A",
            "Val Accuracy": rf_data.get("Val Accuracy", "N/A"),
            "Convergence": "N/A (no training epochs)",
        })
    else:
        print("RF metrics not found — skipping Random Forest row.")

    # 3. Create DataFrame and Table
    df = pd.DataFrame(report_data)
    print("\nComparison Table:")
    if not df.empty:
        print(df.to_markdown(index=False))
    
    # 4. Build extra RF F1 row for the markdown
    rf_f1_str = ""
    if rf_data.get("Val F1 (weighted)"):
        rf_f1_str = f"\n  - **Random Forest Val F1 (weighted)**: {rf_data['Val F1 (weighted)']}"

    # 5. Generate Markdown Report
    n_cnn_kernels = 4  # [3, 5, 7, 9]
    report_md = f"""# E. coli Gene Function Classification Report

## 1. Objective
To classify E. coli protein sequences into functional categories (e.g., 'DNA-binding protein', 'transporter') using multiple model architectures.

## 2. Models Compared
1. **ProteinCausalLM (Transformer)**: A causal language model adapted for classification.
   - Architecture: {CONFIG['n_layers']} layers, {CONFIG['n_heads']} heads, d_model={CONFIG['d_model']}
   - Mechanism: Mean pooling of encoder outputs + Linear Classifier.

2. **ProteinBiLSTM**: A Bidirectional LSTM model.
   - Architecture: {CONFIG['n_layers']} layers, hidden_size={CONFIG['d_model']}
   - Mechanism: Mean pooling of hidden states + Linear Classifier.

3. **ProteinCNN**: A 1D Convolutional Neural Network with multiple kernel sizes.
   - Architecture: {n_cnn_kernels} parallel conv filters (kernels 3, 5, 7, 9), d_model={CONFIG['d_model']}
   - Mechanism: Multi-scale conv → global max pooling → concatenation → Linear Classifier.

4. **Random Forest**: A feature-based tree ensemble baseline.
   - Features: 1-mer, 2-mer, and 3-mer frequency vectors + log sequence length.
   - Algorithm: {rf_data.get('n_estimators', 300)} trees with class-weight balancing (scikit-learn).

## 3. Results Summary

{df.to_markdown(index=False) if not df.empty else "No results found."}{rf_f1_str}

## 4. Technical Details
- **Dataset**: E. coli genomes (TaxID 562).
- **Task**: Multi-class classification of top {CONFIG.get('top_n_classes', 15)} gene products.
- **Loss Function**: CrossEntropyLoss.
- **Optimizer**: AdamW.

## 5. Observations
- **Transformer**: Captures long-range dependencies but requires more compute.
- **BiLSTM**: Efficient for sequences, good at local context.
- **CNN**: Fast training, captures local sequence motifs (k-mer-like patterns) via multiple kernel sizes.
- **Random Forest**: Interpretable baseline using hand-crafted k-mer features; useful lower-bound reference.

## 6. Convergence
(Refer to TensorBoard logs for detailed loss curves)
"""

    with open("model_comparison_report.md", "w") as f:
        f.write(report_md)
        
    print("\nReport saved to model_comparison_report.md")

if __name__ == "__main__":
    generate_report()
