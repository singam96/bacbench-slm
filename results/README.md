# AMR results (ESM-2 35M embeddings + trained heads)

Genome-level resistance prediction for **37 antibiotics** on **6,055 labelled E. coli genomes**
(taxid 562). Each genome is a set of ~4,700 unique proteins; every protein is embedded once with
`facebook/esm2_t12_35M_UR50D` (mean-pooled, 480 dims) and a small trained head reads all of them.
Code: `embed.py`, `order.py`, `amr_embed.py`. Raw-sequence baseline: `amr.py`.

## Results

Macro over the 21 antibiotics with at least 10 resistant validation genomes. 605 validation
genomes (`split.json`: 5,450 train / 605 val, random, seed 42).

| Head | AUROC | AUPRC | Best epoch |
|---|---|---|---|
| attn (gated attention over proteins, per antibiotic) | 0.931 | 0.786 | 30 |
| cnn (1D conv along the genome) | 0.925 | 0.798 | 23 |
| transformer (self-attention over proteins) | 0.923 | 0.772 | 23 |
| bilstm (after 10 more epochs, `cont10`) | 0.912 | 0.753 | +10 |
| bilstm (30 epochs) | 0.904 | 0.740 | 30 |
| mean (mean over proteins + linear) | 0.843 | 0.621 | 30 |
| *raw-sequence CNN, 64 random proteins per genome* | *0.59* | *0.34* | *8* |

## Read this before citing the numbers

- **Single runs, one seed each.** Differences of about 0.01 between heads are within noise.
- **No held-out test set.** The best epoch was chosen on the same validation genomes the scores
  are reported on, so the scores are slightly optimistic.
- **Random split, not lineage-aware.** Near-identical strains can sit on both sides of the split.
- bilstm, attn and mean were still improving at the 30-epoch cap.
- No gene presence/absence (or other simple) baseline has been run yet.

## Files

| File | What |
|---|---|
| `head_esm2_35m_{transformer,cnn,bilstm,attn,mean}.pt` | Best-epoch head weights (state dicts) |
| `head_esm2_35m_cont10_bilstm.pt` | BiLSTM after 10 more epochs |
| `amr_metrics_esm2_35m_*.json` | Macro and per-antibiotic AUROC / AUPRC for each head |
| `amr_metrics_rawseq_cnn_64proteins.json` | Raw-sequence CNN baseline |
| `amr_summary_esm2_35m.json` | Macro metrics for the five heads |
| `heads_training_log.txt`, `bilstm_cont.log` | Per-epoch training logs |
| `split.json` | Train/val genome names and the antibiotic order (output index = this order) |

Not included: the embeddings (`emb.f16`, 1.3 GB; regenerate with `embed.py`, about 30 min on an
A10) and the ESM-2 weights (public on Hugging Face).

## Loading a head

```python
import json, torch
from amr_embed import SeqHead, MILHead

antibiotics = json.load(open("results/split.json"))["antibiotics"]   # 37 outputs, in this order
head = SeqHead(480, len(antibiotics), "cnn")        # or "transformer", "bilstm"
# head = MILHead(480, len(antibiotics), mode="attn")  # or mode="mean"
head.load_state_dict(torch.load("results/head_esm2_35m_cnn.pt", map_location="cpu"))
head.eval()
# input: x (B, N, 480) ESM-2 embeddings of a genome's proteins, mask (B, N) bool
# probs = torch.sigmoid(head(x, mask))   # (B, 37) probability of resistance
```

`cnn` and `bilstm` expect proteins in genome order (`order.py`); `transformer`, `attn` and `mean`
treat the genome as a set.
