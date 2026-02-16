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
    
    # DL Models
    for model_type in ["transformer", "bilstm"]:
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

    # 3. Create DataFrame and Table
    df = pd.DataFrame(report_data)
    print("\nComparison Table:")
    if not df.empty:
        print(df.to_markdown(index=False))
    
    # 4. Generate Markdown Report
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

## 3. Results Summary

{df.to_markdown(index=False) if not df.empty else "No results found."}

## 4. Technical Details
- **Dataset**: E. coli genomes (TaxID 562).
- **Task**: Multi-class classification of top {CONFIG.get('top_n_classes', 15)} gene products.
- **Loss Function**: CrossEntropyLoss.
- **Optimizer**: AdamW.

## 5. Observations
- **Transformer**: Captures long-range dependencies but requires more compute.
- **BiLSTM**: Efficient for sequences, good at local context.

## 6. Convergence
(Refer to TensorBoard logs for detailed loss curves)
"""

    with open("model_comparison_report.md", "w") as f:
        f.write(report_md)
        
    print("\nReport saved to model_comparison_report.md")

if __name__ == "__main__":
    generate_report()
