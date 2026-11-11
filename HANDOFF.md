# Handoff — BacBench-SLM

Last updated: 2026-11-11 · Branch `main` · HEAD `fb11352` (pushed to `origin/main`)

## What this project is

A research benchmark on the Hugging Face BacBench antibiotic-resistance corpus
(`macwiatrak/bacbench-antibiotic-resistance-protein-sequences`), E. coli only (taxid 562). It now
has two tasks:

| Task | What is predicted | Status |
|---|---|---|
| **AMR (new, main)** | resistant / susceptible for 37 antibiotics, per **genome** | working, results in `results/` |
| Gene function (original) | the `product` label of a single **protein** (top-15 strings) | working, but weak framing (see Known issues) |

The original project was titled "AMR prediction" but only predicted gene function and never loaded
resistance labels. That gap is now closed by the AMR task.

## Current results (AMR)

Macro over the 21 antibiotics with at least 10 resistant validation genomes. 6,055 labelled genomes,
random genome split (seed 42): 5,450 train / 605 validation.

| Model | Input | AUROC | AUPRC |
|---|---|---|---|
| Attention head | ESM-2 35M vectors of **all** proteins | 0.931 | 0.786 |
| CNN head | same | 0.925 | 0.798 |
| Transformer head | same | 0.923 | 0.772 |
| BiLSTM head (+10 epochs) | same | 0.912 | 0.753 |
| Mean-pool head (baseline) | same | 0.843 | 0.621 |
| Raw-sequence CNN (`amr.py`) | 64 random proteins per genome | 0.593 | 0.340 |

Reading it: seeing every protein (via a pretrained encoder) is what matters (0.59 → ~0.92). The
top three heads are within noise of each other. Per-antibiotic numbers, weights, logs and the
split are in `results/`; a 5-page PDF is `results/AMR_results_report.pdf`.

**Do not quote these as final.** Single runs, one seed; best epoch chosen on the same validation
genomes the scores come from (no held-out test set); random split, so near-identical strains can sit
on both sides; BiLSTM / attention / mean were still improving at the 30-epoch cap; no simple
baseline has been run.

## How the AMR pipeline works

```
protein sequences --embed.py--> unique proteins (1.35M of 29M occurrences) --ESM-2 35M--> 480-d vectors
                                   |
genome = set of ~4,700 vectors --amr_embed.py (Transformer / CNN / BiLSTM / attention / mean head)--> 37 probabilities
```

- Labels: `binary_labels.csv` in the dataset repo (`genome_name` join, 1 = resistant, 0 = susceptible,
  blank = not tested; blanks are masked out of the loss). 37 antibiotics, 6,055 of the 7,211 E. coli
  genomes have at least one label.
- Loss: masked BCE with per-antibiotic `pos_weight` (capped at 10). Metric: per-antibiotic AUROC/AUPRC.
- Why embeddings: encoding all ~4,700 proteins per genome end to end is too expensive (the 64-protein
  cap in `amr.py` hid most resistance genes). De-duplicating cuts the work ~20x.
- `cnn` and `bilstm` heads read proteins in genome order (`order.py`); `transformer`, `attn`, `mean`
  treat the genome as a set.

## Repository map

| File | Purpose |
|---|---|
| `config.py` | Single `CONFIG` dict. New `amr` section (labels file, batch size, pooling, ...) |
| `amr.py` | AMR end-to-end baseline: genome dataset, `GenomeAMRClassifier`, `LitAMR`, genome-level k-mer Random Forest, `run_amr()` |
| `embed.py` | De-duplicate proteins, embed each unique one with ESM-2 → `emb.f16` memmap (resumable) |
| `order.py` | Per-genome protein ids in genome order → `genome_order.npz` |
| `amr_embed.py` | The five heads over cached embeddings; `--resume` / `--save-tag` to continue training |
| `train.py` | Entry point. `--task gene_function` (default, original) or `--task amr` (+ `--models`, `--epochs`, `--proteins`, `--batch-size`, `--limit-batches`, ...) |
| `dataset.py`, `data_module.py`, `model.py`, `lit_module.py` | Original gene-function pipeline; `model.py` encoders are reused by `amr.py` |
| `create_report.py`, `create_html_report.py` | Reports for the **gene-function** task only (do not read AMR metrics) |
| `inspect_labels.py`, `inspect_ds.py`, `count_ecoli.py`, `dataset_sample.json`, `label_out.txt` | Dataset inspection helpers / outputs |
| `Dockerfile`, `run.bat` | Local CUDA 12.8 image (Windows host); not used for the Vast.ai runs |
| `results/` | Head weights (`*.pt`), metrics JSON, logs, `split.json`, README, PDF report |
| `requirements.txt` | Now includes `transformers` (needed by `embed.py`) |

Not in git: embeddings (`emb.f16` 1.3 GB, `genomes.npz`, `genome_order.npz`, `meta.json`) — saved
locally at `D:\Projects\Python\bacbench-slm-data\emb\`; `.gitignore` ignores `emb/`, `*.f16`, `*.npz`,
`*.pkl`, `checkpoints_amr/`, and `*.pt` except `results/*.pt`.

## How to reproduce (one GPU, ~22 GB is enough)

Local machine has no GPU workflow; runs were done on Vast.ai (below). On any CUDA box with
`pip install -r requirements.txt`:

```bash
export HF_HOME=/workspace/.hf_home            # dataset cache is ~70 GB (32 GB download + 38 GB arrow)
python train.py --task amr --models cnn --test   # smoke test (writes checkpoints_amr/test_run/)
python train.py --task amr --models cnn          # raw-sequence baseline; also writes checkpoints_amr/split.json
python embed.py --stage all                      # count + embed, ~32 min on an A10 -> /workspace/emb
python order.py                                  # genome-order ids
python amr_embed.py --modes transformer,cnn,bilstm,attn,mean   # ~35 min total on an A10
# continue a head: python amr_embed.py --modes bilstm --resume --epochs 10 --lr 5e-4 --save-tag esm2_35m_cont10
```

**Gotcha:** `amr_embed.py` reads `<out>/split.json` (written by `amr.py`'s `run_amr`). On a fresh
checkout copy `results/split.json` to `checkpoints_amr/split.json` first, or the head script fails.

Measured on an A10 (22 GB): raw-seq CNN 110 s/epoch (0.6 GiB peak); ESM-2 35M embedding of 1.35M
proteins in 32 min; heads (30 epochs): Transformer ~10 min, CNN ~2 min, BiLSTM ~18 min, attn ~3.5 min,
mean ~2 min. For the raw-seq BiLSTM and Transformer only a 60-batch probe exists (23 s / 31 s, peak
3.8 / 7.5 GiB at 2 genomes x 64 proteins); the ~10 / ~13 min per epoch figures are extrapolated from it.

## Vast.ai notes (what cost time)

- Whole AMR effort used ~$1 of GPU time (A10, $0.274/hr, ~3.3 h). **No instance is running.** Credit
  was $7.07 at last check. SSH key: `~/.ssh/vast_key`.
- Template "PyTorch (Vast) CUDA 12"; python is `/venv/main/bin/python` (system python has no torch).
  Install: `pytorch-lightning datasets scikit-learn pandas pyarrow matplotlib tabulate transformers`.
- The public repo can be `git clone`d on the instance; copy uncommitted files with `scp`.
- Image pulls can take 15+ minutes on slow hosts. One RTX 3090 sat on one layer for 25+ minutes and was
  destroyed; pick hosts with fast network (the Illinois A10, ~900 Mbps, was running in ~4 min).
- Launch long jobs with `(setsid nohup cmd > log 2>&1 < /dev/null &)`. `ssh host 'cmd &'` keeps the ssh
  session attached. Never `pkill -f "<text in your own ssh command>"` — it kills its own shell.
- Disk: dataset cache is 70 GB of the 100 GB default disk; 30 GB free.

## Dataset facts (verified)

- 26,052 genomes in total (one row each), 403 distinct taxids. Actual download is **32 GB** (the
  dataset viewer's 4.6 GB figure covers a partial index only).
- taxid 562: **7,211** genomes. Another 42 genomes sit under per-strain "Escherichia coli ..." taxids
  and are *not* included by the exact-match filter (<1%); 33 Shigella genomes are also excluded.
- 29.0M protein occurrences in the 6,055 labelled genomes, 1.35M unique sequences, ~4,758 unique
  proteins per genome on average, longest genome 5,818. 18.8% of proteins exceed 512 aa and are
  truncated to 512 for embedding.
- Label table: ampicillin (43.8% resistant) and tetracycline (40.3%) are best balanced; meropenem,
  amikacin, azithromycin have <50 resistant genomes and are not averaged.

## Known issues / open items

1. **Evaluation is not final** — see the caveats above. Needs a held-out test split, ideally
   lineage-aware, plus several seeds.
2. **No simple baseline on the full genome** (gene presence/absence + logistic regression). Without it
   we cannot say how much of 0.92 needs a neural network.
3. `amr.py` end-to-end results exist only for the CNN. The raw-seq BiLSTM and Transformer runs were
   stopped; the genome-level Random Forest in `amr.py` only ran in the smoke test (64 proteins/genome).
4. `amr_embed.py` depends on `split.json` from `amr.py` (see Gotcha). A fallback that recomputes the
   same seeded split from the genome list would remove this.
5. The two report scripts do not read AMR metrics; `results/AMR_results_report.pdf` was built by an
   ad-hoc matplotlib script (not in the repo).
6. 42 E. coli strain taxids are missed by the species filter (see Dataset facts).
7. **Gene-function task** (unchanged): top-15 `product` strings cover only ~8.8% of proteins (rest are
   `-100`), classes are noisy stringified lists, so its accuracies (RF 0.969, CNN 0.924, BiLSTM 0.878,
   Transformer 0.854) look better than they are. Its checkpoints/logs are gone, so
   `create_html_report.py` cannot regenerate its report without retraining. Metrics are parsed from
   checkpoint filenames.
8. **Decks** (`AMR Prediction- Bi-LSTM vs Transformers.pptx`, gitignored; `SECOND DOCTORAL COMMITTEE
   MEETING.pptx`, untracked) still describe the gene-function task (slides 4-13) with its numbers. They
   have **not** been updated for the AMR results. `todo.md` still lists the visual fit check as pending.
9. No tests, no CI, no linter config.

## Recommended next steps

| Step | Why | Cost |
|---|---|---|
| Held-out test split (+ lineage-aware option), re-run heads with early stopping and 3+ seeds | makes the numbers defensible | ~1-2 h GPU, <$1 |
| Gene presence/absence baseline (cluster proteins, logistic regression) | the standard AMR baseline; shows what the neural heads add | a few hours incl. clustering |
| Update the committee deck to the AMR task and results | decks are out of sync with the code | manual |
| Optional: larger ESM-2 (150M / 650M); extend `create_report*.py` to AMR metrics; add the missing 42 strain taxids | quality / completeness | varies |

## Verification commands

```bash
python -m py_compile amr.py embed.py order.py amr_embed.py train.py config.py
python -c "import json; d=json.load(open('results/amr_summary_esm2_35m.json')); print({k: round(v['auroc'],3) for k,v in d.items()})"
python -c "import json; print(len(json.load(open('results/split.json'))['antibiotics']))"   # 37
git log --oneline -4          # fb11352 ... eaf7606 (AMR code) ... 8f64a78
```
