# Dataset acquisition and algorithm-ready inputs

Run commands from the repository root. Install the controller dependencies
first, as described in the main README. Downloaded observations retain their
upstream terms; model input arrays and saved scores stay in ignored result
directories. Public availability alone does not establish redistribution
permission. This repository includes the bounded generated examples below
and obtains real observations from their original hosts.

## Included generated examples

| Directory | Contents | Approximate size |
| --- | --- | ---: |
| `data/contextual/` | Project-generated normal training and labeled test trajectories, with metadata | 31 MiB |
| `data/distribution_preserving/` | Project-generated trajectory reorderings and normal training, with metadata | 31 MiB |
| `data/gutentag_extension_v1/` | 18 generated three-channel cases, separate normal training/test CSVs, configuration, seeds, hashes, and generator environment | 18 MiB |

The largest included CSV is approximately 21 MiB, below GitHub's
[100 MiB per-file limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
The contextual and distribution-preserving files originate from
`generate_pendulum_dataset.py`. Read their `feature_columns` metadata and use
only observed `x_*` columns as inputs. `entity_id` and `time` describe sequence
membership and ordering. `is_anomaly` is an evaluation label; latent simulator
columns are diagnostic information.

To generate a new distribution-preserving split:

```bash
.venv/bin/python generate_pendulum_dataset.py \
  --out-dir /tmp/dynsphere-generated-example \
  --name pendulum_distribution_preserving \
  --anomaly-profile distribution_preserving
```

Choose a new output directory to preserve an earlier generated copy.

The GutenTAG examples were generated with
[TimeEval/GutenTAG](https://github.com/TimeEval/GutenTAG), version 1.5.0,
using this project's configuration. The upstream generator is MIT-licensed;
its authors and license are acknowledged in the included dataset notice.
The examples use six mechanisms and generation seeds 101, 102, and 103.
Each `test.csv` has a paired `train_no_anomaly.csv`; `timestamp` and
`is_anomaly` are excluded from observed model features. File hashes and
relative paths are recorded in `native_manifest.json`.

To regenerate in an isolated environment, using a new output directory:

```bash
python3.11 -m venv benchmarks/official_baselines/.envs/gutentag
benchmarks/official_baselines/.envs/gutentag/bin/python -m pip install timeeval-gutenTAG==1.5.0
benchmarks/official_baselines/.envs/gutentag/bin/python \
  -m benchmarks.official_baselines.prepare_gutentag_extension \
  --output-dir /tmp/dynsphere-gutentag-generated
```

The loader accepts the resulting `native_manifest.json`. See the
[extension guide](../benchmarks/official_baselines/EXTENSION_EVALUATION.md)
for the corresponding command and the distinction between generation and
detector seeds.

## BATADAL and HAI-20.07

```bash
.venv/bin/python benchmarks/real_world_datasets/download_real_world_datasets.py --dataset batadal
.venv/bin/python benchmarks/real_world_datasets/download_real_world_datasets.py --dataset hai_20_07
```

Expected files are:

```text
benchmarks/real_world_datasets/data/batadal/BATADAL_dataset03.csv
benchmarks/real_world_datasets/data/batadal/BATADAL_dataset04.csv
benchmarks/real_world_datasets/data/hai_20_07/train1.csv.gz
benchmarks/real_world_datasets/data/hai_20_07/test1.csv.gz
```

BATADAL uses `ATT_FLAG` as its label. HAI uses `attack`. The loaders select
matching numeric feature columns and exclude timestamps and labels. The
held-out protocol divides normal training into 80% fitting and 20% calibration
and keeps the released test file separate. Scaling and imputation statistics
are learned on fitting rows. Historical commands that subsample observations
or score training rows describe a different protocol.

Sources and required attribution:

- [BATADAL data](https://www.batadal.net/data.html): Taormina et al.,
  *The Battle of the Attack Detection Algorithms: Disclosing Cyber Attacks
  on Water Distribution Networks*, 2018. The site footer links CC BY 4.0.
  Obtain the data directly to retain the source's notices and release context.
- [HAI dataset](https://github.com/icsdataset/hai): Shin, Lee, Yun, and Kim,
  *HAI 1.0: HIL-Based Augmented ICS Security Dataset*, 2020. The upstream
  License section specifies CC BY-SA 4.0; its dataset metadata also contains
  a CC BY 4.0 entry. We retain source acquisition and its attribution notices.

If an automated URL is unavailable, download the named release files from
the source and place them at the exact paths above. A successful download
must contain CSV data, with finite binary labels and the expected features.
An HTML error page is not a usable dataset.

## TSB-AD-M: MSL, SMAP, SMD, Exathlon, and GHL preparations

```bash
.venv/bin/python benchmarks/tsb_ad_m/download_tsb_ad_m.py
```

The script downloads and extracts the archive into:

```text
benchmarks/tsb_ad_m/data/raw/TSB-AD-M.zip
benchmarks/tsb_ad_m/data/TSB-AD-M/
```

If the host rejects the first download attempt:

```bash
.venv/bin/python benchmarks/tsb_ad_m/download_tsb_ad_m.py --method curl --force
```

The fixed evaluation selection is
`benchmarks/official_baselines/manifests/TSB-AD-M-Eva.csv`. A source filename
contains its declared training-prefix length as `_tr_N_`. `Label` is the
binary evaluation column. For the current held-out comparison, a clean
prefix is divided into fitting and normal calibration, and the suffix is
the test sequence. Contaminated training prefixes are recorded as incompatible.
The matched deep/local comparison excludes 100 observations at both test
boundaries. The Exathlon/GHL extension has its own boundary rule, documented
in the extension guide. Do not substitute the original NASA/SMD releases
for these prepared files when reproducing a TSB-AD comparison.

The local TSB-AD copy is approximately 3 GiB including its archive. It is
excluded from Git. Acquire it from [TheDatum/TSB-AD](https://github.com/TheDatumOrg/TSB-AD)
and retain the original benchmark and source-family citations and terms.
TSB-AD source-family preparations and their original datasets are distinct
experimental inputs.

## Source-native Exathlon

The separate execution-disjoint study reads the pinned raw
[Exathlon release](https://github.com/exathlonbenchmark/exathlon), commit
`4101f6087f902fa150e392b65c957976faa84e40`. Its loader verifies Git blob
identities, handles the split-volume application-4 archive, derives 19 Spark
features, and writes finite prepared arrays and sequence-boundary masks.
No model window or propagation crosses a timestamp gap or execution boundary.

Acquire the complete release inventory, then prepare inputs:

```bash
mkdir -p results/exathlon_native_frozen_20261025
curl --fail --location \
  'https://api.github.com/repos/exathlonbenchmark/exathlon/git/trees/4101f6087f902fa150e392b65c957976faa84e40?recursive=1' \
  --output results/exathlon_native_frozen_20261025/release_tree.json
.venv/bin/python -m research.exathlon_native.run prepare \
  --tree results/exathlon_native_frozen_20261025/release_tree.json
```

This selects all disturbed executions in application/rate groups with at
least two normal executions. At the pinned release it selects applications
1, 2, 4, and 5, rate 100,000, and five test executions. The first two numeric
normal execution identifiers supply fitting and calibration. Raw archives
total approximately 497 MiB before extraction, so they are excluded from Git.
The dataset's [CC BY-NC-SA 4.0 terms](https://github.com/exathlonbenchmark/exathlon#data-license)
remain applicable; the repository's software license is separate.

Preparation downloads archives and verifies them; it does not start model
training or submit jobs. The [native execution guide](../research/exathlon_native/README.md)
describes external-environment setup, compatibility checks, freezing, local
execution, GPU runners, and metric evaluation. VUS uses only eligible
within-segment support, which can differ from pointwise support.

## PowerSystem

```bash
.venv/bin/python benchmarks/real_world_datasets/download_real_world_datasets.py --dataset power_system
```

Extract the downloaded archive using its upstream README. For the
source-file-disjoint diagnostic, preserve the individual binary files as:

```text
benchmarks/real_world_datasets/data/power_system/binary/data1.csv
...
benchmarks/real_world_datasets/data/power_system/binary/data15.csv
```

The diagnostic uses normal observations from files 1--9 for fitting, normal
observations from files 10--12 for calibration, and files 13--15 for testing.
The `marker` column supplies labels. The historical runner instead expects
`combined_binary.csv` and uses a different protocol. Preserve source-file
boundaries; a combined file cannot reconstruct the grouped diagnostic.

The public [ICS dataset source](https://ece.uah.edu/~thm0009/icsdatasets/)
and its archive README provide acquisition and citation details. No explicit
redistribution license was established for our copy, so it is excluded from
Git. Temporal continuity and scenario independence remain unverified, and
these results are reported as a sampled-order diagnostic.

## Reading generated artifacts

Case loaders write normal model inputs separately from evaluation labels.
Frozen campaigns additionally record source hashes, feature ordering, split
boundaries, model configuration, seeds, and score coverage. Loaders can reject
a missing source, a contaminated prefix, or a nonfinite observation. Retain
these records and do not replace cases based on their measured accuracy.
Historical summaries and frozen score arrays are separate artifacts and
must be supplied to scripts that explicitly require them.
