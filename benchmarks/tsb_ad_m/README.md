# TSB-AD-M Validation

This folder keeps real multivariate time-series validation separate from the synthetic DynSphere benchmark.

TSB-AD-M is the multivariate split of TSB-AD. The official TSB-AD README describes the CSV format as feature columns followed by a `Label` column, where labels are used for evaluation and the preceding columns are the time-series variables.

## Download

```bash
python3 benchmarks/tsb_ad_m/download_tsb_ad_m.py
```

This downloads `https://www.thedatum.org/datasets/TSB-AD-M.zip` into `benchmarks/tsb_ad_m/data/raw/` and extracts it into `benchmarks/tsb_ad_m/data/TSB-AD-M/`.

The data are not committed to this workspace.

## Run

Evaluate a small smoke subset. Preferred CLI method names are `dynsphere`,
`dynsphere_ae`, and `dynsphere_pp`; legacy aliases remain accepted for
compatibility with older result scripts.

```bash
python3 benchmarks/tsb_ad_m/run_tsb_ad_m.py --limit 5 --methods dynsphere sequence_ae dynsphere_ae
```

Run more files:

```bash
python3 benchmarks/tsb_ad_m/run_tsb_ad_m.py --limit 50 --fit-mode all --methods dynsphere sequence_ae dynsphere_ae
```

Outputs are written to:

- `benchmarks/tsb_ad_m/results/tsb_ad_m_results.csv`
- `benchmarks/tsb_ad_m/results/tsb_ad_m_summary.json`

## Fit Modes

- `all`: fully unsupervised; fit on the full unlabeled series and score the same series. This follows the common unsupervised TSAD setup, but models may absorb anomalies during fitting.
- `prefix`: fit only on the contiguous prefix before the first anomaly. This is realistic when a normal warm-up period exists.
- `oracle-clean`: fit only on rows where `Label == 0`. This uses labels during training and should be treated as a diagnostic upper-bound, not a fair unsupervised result.

## Methods

- `dynsphere`: DynSphere detector
- `sequence_ae`: windowed sequence autoencoder
- `dynsphere_ae`: DynSphereAE, robust fusion of sequence autoencoder and DynSphere
- `dynsphere_pp`: DynSphere++ score-level ensemble
- PyOD baselines: `lof`, `knn`, `iforest`, `hbos`, `copod`, `pca`

This runner reports AUROC and AUPRC. Official TSB-AD studies emphasize VUS-style metrics as well; this folder is a lightweight integration layer, not a replacement for the official benchmark harness.

Run with official VUS metrics when the `vus` package is available:

```bash
uv run --with numpy --with pandas --with scikit-learn --with pyod --with torch --with vus \
  python benchmarks/tsb_ad_m/run_tsb_ad_m.py \
  --limit 5 \
  --fit-mode all \
  --methods dynsphere sequence_ae dynsphere_ae hbos copod \
  --include-vus \
  --out-dir benchmarks/tsb_ad_m/results_limit5_all_vus
```

VUS metrics are optional because the official package pulls a large dependency stack.
