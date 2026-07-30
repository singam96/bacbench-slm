import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import glob
from config import CONFIG

def generate_report():
    print(f"\n{'='*20}\nGenerating Model Comparison Report\n{'='*20}")
    
    report_data = []
    
    # 1. Gather Deep Learning Model Metrics (Transformer, BiLSTM)
    # These are stored in checkpoint dirs by PyTorch Lightning or we need to parse logs.
    # Actually, we didn't explicitly save a "metrics.json" for DL models in train.py, 
    # but we have TensorBoard logs or checkpoints.
    # For simplicity, let's assume we want to report the BEST validation metrics found in filenames 
    # or we should have saved them.
    # Let's updated train.py to save metrics? No, too late/risky to re-run everything.
    # We can parse the checkpoint filenames for val_loss.
    
    checkpoint_dir = CONFIG["checkpoint_dir"]
    
    # DL Models (Transformer, BiLSTM, CNN)
    for model_type in ["transformer", "bilstm", "cnn"]:
        fold_dirs = glob.glob(os.path.join(checkpoint_dir, f"{model_type}_fold_*"))
        
        for fd in fold_dirs:
            # Find best checkpoint
            ckpts = glob.glob(os.path.join(fd, "*.ckpt"))
            if not ckpts:
                continue
                
            # Filename format: "best-epoch=XX-val_loss=YY.ckpt" or similar
            # My train.py used: filename="epoch_{epoch:04d}_val_{val_loss:.4f}"
            
            best_ckpt = ckpts[0] # Assume one if save_top_k=1
            # Parse metrics from filename
            val_loss = None
            val_acc = None
            filename = os.path.basename(best_ckpt)
            
            # Format: epoch_0001_loss_0.4523_acc_0.8912.ckpt
            try:
                # Simple parsing by splitting
                parts = filename.replace(".ckpt", "").split("_")
                
                # Iterate to find keys
                for i, part in enumerate(parts):
                    if part == "loss" and i+1 < len(parts):
                        val_loss = float(parts[i+1])
                    if part == "acc" and i+1 < len(parts):
                        val_acc = float(parts[i+1])
                    if part == "val" and i+1 < len(parts): # Fallback for old format
                        # epoch_0001_val_0.4523
                        try:
                            val_loss = float(parts[i+1])
                        except:
                            pass
            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")
            
            report_data.append({
                "Model": model_type,
                "Type": "Deep Learning",
                "Fold": os.path.basename(fd),
                "Val Loss": val_loss,
                "Val Accuracy": val_acc if val_acc is not None else "N/A",
                "Convergence": "See TensorBoard"
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
