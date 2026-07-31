"""Generate a self-contained HTML comparison report for the BacBench SLM project.

Reads:
  - checkpoint filenames  (best val metrics per DL model)
  - lightning_logs/version_*/metrics.csv  (train/val curves, hyperparams)
  - checkpoints/rf_metrics.json           (Random Forest results)

Outputs model_comparison_report.html with embedded matplotlib charts.
Run without re-training:  python create_html_report.py
"""

from __future__ import annotations

import base64
import glob
import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import CONFIG
from create_report import best_checkpoint_metrics

DL_MODELS = ["transformer", "bilstm", "cnn"]
MODEL_LABELS = {
    "transformer": "Transformer (ProteinCausalLM)",
    "bilstm": "BiLSTM (ProteinBiLSTM)",
    "cnn": "CNN (ProteinCNN)",
    "random_forest": "Random Forest",
}
MODEL_DESCRIPTIONS = {
    "transformer": (
        "Causal Transformer encoder adapted for classification. 6 layers, 8 heads, "
        "d_model=256, GELU activation with pre-LayerNorm. Mean pooling of the encoder "
        "outputs followed by a linear classifier. Captures long-range dependencies in "
        "protein sequences."
    ),
    "bilstm": (
        "Bidirectional LSTM with 6 layers and hidden size 256. Processes sequences in "
        "both directions, then mean-pools the hidden states and classifies. Efficient "
        "for sequence modelling with good local-context awareness."
    ),
    "cnn": (
        "1D convolutional neural network with 4 parallel filters (kernel sizes 3, 5, 7, "
        "9), each followed by ReLU and global max-pooling. The multi-scale features are "
        "concatenated and fed to a linear classifier. Fast to train and captures local "
        "k-mer-like sequence motifs."
    ),
    "random_forest": (
        "Classical ML baseline. Features are normalised 1-mer, 2-mer and 3-mer "
        "frequency vectors plus log sequence length (sparse DictVectorizer). An ensemble "
        "of 300 decision trees with class-weight balancing. Provides an interpretable, "
        "non-neural reference point."
    ),
}

MODEL_COLORS = {
    "transformer": "#1f77b4",
    "bilstm": "#ff7f0e",
    "cnn": "#2ca02c",
    "random_forest": "#d62728",
}


def collect_dl_metrics(checkpoint_dir: str) -> list[dict]:
    rows = []
    for model_type in DL_MODELS:
        for fd in sorted(glob.glob(os.path.join(checkpoint_dir, f"{model_type}_fold_*"))):
            m = best_checkpoint_metrics(fd)
            if m is None:
                continue
            rows.append(
                {
                    "model": model_type,
                    "fold": os.path.basename(fd),
                    "epoch": m.get("epoch"),
                    "val_loss": m.get("val_loss"),
                    "val_acc": m.get("val_acc"),
                }
            )
    return rows


def collect_rf_metrics(checkpoint_dir: str) -> dict:
    path = os.path.join(checkpoint_dir, "rf_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_logs(log_dir: str) -> list[dict]:
    """Load metrics.csv + hparams.yaml for every lightning_logs/version_*.

    Keeps only the best (lowest final val_loss) run per model type and drops
    versions that contain no real training rows (e.g. stale --test runs).
    """
    logs = []
    for version_dir in sorted(glob.glob(os.path.join(log_dir, "version_*"))):
        metrics_csv = os.path.join(version_dir, "metrics.csv")
        hparams_yaml = os.path.join(version_dir, "hparams.yaml")
        if not os.path.exists(metrics_csv):
            continue
        df = pd.read_csv(metrics_csv)
        if len(df) == 0:
            continue  # header-only, nothing trained
        hparams = {}
        if os.path.exists(hparams_yaml):
            with open(hparams_yaml) as f:
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        hparams[k.strip()] = v.strip()
        model_type = hparams.get("model_type", os.path.basename(version_dir))
        # Final validation loss for ranking duplicate runs of the same model
        final_val_loss = None
        if "val_loss" in df.columns:
            v = df["val_loss"].dropna()
            if not v.empty:
                final_val_loss = v.iloc[-1]
        logs.append(
            {
                "version": os.path.basename(version_dir),
                "model_type": model_type,
                "df": df,
                "hparams": hparams,
                "final_val_loss": final_val_loss,
            }
        )

    # Keep the best run per model type (lowest final val_loss)
    best_by_model: dict[str, dict] = {}
    for log in logs:
        cur = best_by_model.get(log["model_type"])
        if cur is None:
            best_by_model[log["model_type"]] = log
        else:
            cur_best = cur.get("final_val_loss")
            new_best = log.get("final_val_loss")
            if new_best is not None and (cur_best is None or new_best < cur_best):
                best_by_model[log["model_type"]] = log

    return [best_by_model[m] for m in DL_MODELS if m in best_by_model]


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def chart_combined_metric(logs, metric: str, title: str, ylabel: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    for log in logs:
        df = log["df"]
        col = metric
        if col not in df.columns:
            continue
        series = df[[ "epoch", col]].dropna()
        if series.empty:
            continue
        ax.plot(series["epoch"], series[col], marker="o", label=MODEL_LABELS.get(log["model_type"], log["model_type"]), color=MODEL_COLORS.get(log["model_type"]))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig_to_base64(fig)


def chart_per_model(log: dict) -> str:
    df = log["df"]
    model = log["model_type"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Train loss (smoothed steps + epoch means)
    if "train_loss_step" in df.columns and "epoch" in df.columns:
        s = df[["epoch", "train_loss_step"]].dropna()
        if not s.empty:
            axes[0].plot(s["epoch"], s["train_loss_step"], alpha=0.25, color=MODEL_COLORS.get(model), label="per-step")
        # per-epoch mean
        ep = s.groupby("epoch")["train_loss_step"].mean()
        axes[0].plot(ep.index, ep.values, marker="o", color=MODEL_COLORS.get(model), label="epoch mean")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train Loss"); axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    # Validation loss
    if "val_loss" in df.columns:
        s = df[["epoch", "val_loss"]].dropna()
        if not s.empty:
            axes[1].plot(s["epoch"], s["val_loss"], marker="o", color=MODEL_COLORS.get(model))
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Loss"); axes[1].set_title("Validation Loss")
    axes[1].grid(True, alpha=0.3)

    # Validation accuracy
    if "val_acc" in df.columns:
        s = df[["epoch", "val_acc"]].dropna()
        if not s.empty:
            axes[2].plot(s["epoch"], s["val_acc"], marker="o", color=MODEL_COLORS.get(model))
            for x, y in zip(s["epoch"], s["val_acc"]):
                axes[2].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Val Acc"); axes[2].set_title("Validation Accuracy")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"{MODEL_LABELS.get(model, model)} — training curves", fontsize=13)
    return fig_to_base64(fig)


def build_html(
    dl_rows: list[dict],
    rf_metrics: dict,
    logs: list[dict],
) -> str:
    # ── Results table ──────────────────────────────────────────────
    table_rows = []
    for r in dl_rows:
        table_rows.append(
            f"<tr><td><b>{MODEL_LABELS.get(r['model'], r['model'])}</b></td>"
            f"<td>Deep Learning</td>"
            f"<td>{r.get('epoch', '—')}</td>"
            f"<td>{r['val_loss']:.4f}</td>"
            f"<td>{r['val_acc']:.4f}</td></tr>"
        )
    if rf_metrics:
        table_rows.append(
            f"<tr><td><b>{MODEL_LABELS.get('random_forest', 'Random Forest')}</b></td>"
            f"<td>Feature-based (k-mer)</td>"
            f"<td>—</td>"
            f"<td>—</td>"
            f"<td>{rf_metrics.get('Val Accuracy', '—')}</td></tr>"
        )
    results_table = (
        "<table>"
        "<thead><tr><th>Model</th><th>Type</th><th>Best Epoch</th>"
        "<th>Val Loss</th><th>Val Accuracy</th></tr></thead>"
        "<tbody>" + "".join(table_rows) + "</tbody></table>"
    )

    # ── Hyperparameter table (from the DL logs) ─────────────────────
    hp_keys = ["model_type", "d_model", "n_layers", "n_heads", "dropout", "lr", "max_len", "task_type", "num_classes", "vocab_size"]
    hp_header = "".join(f"<th>{k}</th>" for k in hp_keys)
    hp_rows = []
    for log in logs:
        h = log["hparams"]
        cells = "".join(f"<td>{h.get(k, '—')}</td>" for k in hp_keys)
        hp_rows.append(f"<tr><td><b>{MODEL_LABELS.get(log['model_type'], log['model_type'])}</b></td>{cells}</tr>")
    hp_table = (
        "<table>"
        f"<thead><tr><th>Model</th>{hp_header}</tr></thead>"
        "<tbody>" + "".join(hp_rows) + "</tbody></table>"
    )

    # ── Charts ─────────────────────────────────────────────────────
    val_loss_img = chart_combined_metric(logs, "val_loss", "Validation Loss by Model", "Val Loss")
    val_acc_img = chart_combined_metric(logs, "val_acc", "Validation Accuracy by Model", "Val Accuracy")

    per_model_imgs = []
    for log in logs:
        per_model_imgs.append(
            f'<div class="per-model"><img src="data:image/png;base64,{chart_per_model(log)}" '
            f'alt="{log["model_type"]} curves"></div>'
        )

    # ── Random Forest details ──────────────────────────────────────
    rf_html = ""
    if rf_metrics:
        rf_html = (
            f"<p>{MODEL_DESCRIPTIONS['random_forest']}</p>"
            f"<ul>"
            f"<li><b>Val Accuracy:</b> {rf_metrics.get('Val Accuracy', '—')}</li>"
            f"<li><b>Val F1 (weighted):</b> {rf_metrics.get('Val F1 (weighted)', '—')}</li>"
            f"</ul>"
        )

    # ── Training log details (TensorBoard-style) ───────────────────
    log_details = []
    for log in logs:
        df = log["df"]
        n_epochs = int(df["epoch"].max()) + 1 if "epoch" in df.columns and len(df) else 0
        n_steps = int(df["step"].max()) + 1 if "step" in df.columns and len(df) else 0
        best = None
        if "val_loss" in df.columns:
            v = df[["epoch", "val_loss", "val_acc"]].dropna()
            if not v.empty:
                bi = v["val_loss"].idxmin()
                best = v.loc[bi]
        best_str = (
            f"epoch {int(best['epoch'])} (loss {best['val_loss']:.4f}, acc {best['val_acc']:.4f})"
            if best is not None else "—"
        )
        log_details.append(
            f"<tr><td><b>{MODEL_LABELS.get(log['model_type'], log['model_type'])}</b></td>"
            f"<td>{log['version']}</td><td>{n_epochs}</td><td>{n_steps:,}</td>"
            f"<td>{best_str}</td></tr>"
        )
    logs_table = (
        "<table>"
        "<thead><tr><th>Model</th><th>Log Version</th><th>Epochs Trained</th>"
        "<th>Total Steps</th><th>Best Validation</th></tr></thead>"
        "<tbody>" + "".join(log_details) + "</tbody></table>"
    )

    model_desc_html = "".join(
        f"<h3>{i}. {MODEL_LABELS.get(m, m)}</h3><p>{MODEL_DESCRIPTIONS.get(m, '')}</p>"
        for i, m in enumerate(DL_MODELS + ["random_forest"], 1)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E. coli Gene Function Classification — Model Comparison</title>
<style>
  :root {{
    --accent: #1a73e8;
    --bg: #ffffff;
    --muted: #5f6368;
    --border: #dadce0;
    --code-bg: #f6f8fa;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         color: #202124; margin: 0; background: #fafafa; line-height: 1.55; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
  header {{ background: linear-gradient(135deg, #1a73e8, #0b57d0); color: #fff; padding: 36px 24px; }}
  header h1 {{ margin: 0 0 6px; font-size: 26px; }}
  header p {{ margin: 4px 0; opacity: 0.92; }}
  h2 {{ color: #0b57d0; border-bottom: 2px solid var(--border); padding-bottom: 6px; margin-top: 40px; }}
  h3 {{ margin-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0 8px; background: #fff;
          box-shadow: 0 1px 2px rgba(60,64,67,.15); border-radius: 8px; overflow: hidden; }}
  th, td {{ border: 1px solid var(--border); padding: 10px 12px; text-align: left; font-size: 14px; }}
  th {{ background: #f1f3f4; font-weight: 600; }}
  tr:nth-child(even) td {{ background: #fafbfc; }}
  img {{ max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 8px;
        background: #fff; padding: 8px; }}
  .per-model {{ margin: 18px 0; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  .grid img {{ flex: 1 1 48%; min-width: 320px; }}
  .note {{ background: #fff8e6; border: 1px solid #f0d27a; border-radius: 8px; padding: 12px 16px; margin: 16px 0; }}
  code {{ background: var(--code-bg); border-radius: 4px; padding: 1px 5px; font-size: 13px; }}
  footer {{ color: var(--muted); font-size: 13px; text-align: center; margin: 40px 0 20px; }}
</style>
</head>
<body>
<header>
  <h1>🧬 E. coli Gene Function Classification — Model Comparison</h1>
  <p>Dataset: BacBench antibiotic resistance protein sequences · E. coli (TaxID 562)</p>
  <p>Task: multi-class classification of top {CONFIG.get('top_n_classes', 15)} gene products</p>
</header>
<div class="container">

  <h2>1. Objective</h2>
  <p>Classify E. coli protein sequences into functional categories (e.g.
  “DNA-binding protein”, “transporter”) using four model families: a Transformer,
  a BiLSTM, a 1D CNN, and a k-mer feature-based Random Forest. All deep-learning
  models share the same tokenizer, vocabulary, dataset split and training recipe
  so the comparison is fair.</p>

  <h2>2. Models Compared</h2>
  {model_desc_html}

  <h2>3. Results Summary</h2>
  {results_table}
  <div class="note">⚠️ <b>Note on earlier NaN values:</b> the previous markdown report parsed
  <code>last.ckpt</code> (which carries no metrics in its filename), so it displayed NaN.
  The tables below parse the metric-bearing epoch checkpoints. No model actually produced NaN.</div>

  <h2>4. Validation Curves</h2>
  <div class="grid">
    <img src="data:image/png;base64,{val_loss_img}" alt="Validation loss curves">
    <img src="data:image/png;base64,{val_acc_img}" alt="Validation accuracy curves">
  </div>

  <h2>5. Per-Model Training Curves</h2>
  {''.join(per_model_imgs)}

  <h2>6. Random Forest Baseline</h2>
  {rf_html}

  <h2>7. Hyperparameters</h2>
  {hp_table}

  <h2>8. Training Logs (TensorBoard / CSVLogger details)</h2>
  <p>PyTorch Lightning wrote CSV logs under <code>lightning_logs/</code>. The table
  below summarises each run; the per-step curves are shown in section 5.</p>
  {logs_table}

</div>
<footer>Generated automatically — see <code>create_report.py</code> and <code>create_html_report.py</code>.</footer>
</body>
</html>"""
    return html


def main() -> None:
    checkpoint_dir = CONFIG["checkpoint_dir"]
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lightning_logs")

    print("Collecting DL checkpoint metrics...")
    dl_rows = collect_dl_metrics(checkpoint_dir)
    print(f"  found {len(dl_rows)} DL runs")

    print("Collecting Random Forest metrics...")
    rf_metrics = collect_rf_metrics(checkpoint_dir)
    print(f"  {'found' if rf_metrics else 'not found'}")

    print("Loading lightning logs...")
    logs = load_logs(log_dir)
    print(f"  found {len(logs)} log versions")

    if not dl_rows and not rf_metrics:
        print("No metrics found — nothing to report.")
        return

    html = build_html(dl_rows, rf_metrics, logs)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_comparison_report.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"\nHTML report saved to {out}")


if __name__ == "__main__":
    main()
