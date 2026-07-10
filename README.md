# tabular-data-to-images

Experiments converting tabular data into image representations for CNN-based classification, compared against traditional tabular baselines, across three datasets: network traffic (DDoS), COVID risk, and earthquake soil classification (PSA).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10 (uv will fetch an isolated Python 3.10 build automatically, so a system install isn't required).

```
uv sync
```

This creates `.venv`, installs every dependency pinned in `pyproject.toml`/`uv.lock` (including the git dependencies `pyDeepInsight` and `tabpfn-extensions`), and installs this repo itself as an editable package (`tabular_data_to_images`).

`torch`/`torchvision` are configured by default to install the CUDA 12.6 build (`pyproject.toml`'s `[tool.uv.sources]`/`[[tool.uv.index]]`), matching an NVIDIA GPU setup. If you're on a CPU-only machine, override after syncing:

```
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

(Note: `uv run`/`uv sync` will reinstall the CUDA build again the next time they run, since that's what `pyproject.toml` declares — pass `--no-sync` to `uv run` if you need the CPU override to stick across runs.)

**Optional:** `mpi4py` is not installed by default. It's only needed if you want to regenerate the REFINED feature-to-pixel mappings from scratch via `refined/mpiHill_UF.py` — see "Data" below, these mappings are not committed.

## Data

**Nothing under `dataset/`, `datasets/`, or `trained_models/` is committed to this repo** (CSVs, images, model checkpoints, and pickles are all gitignored). Everything needed to run the experiments must be placed locally first:

- **`dataset/`** (repo root) — the processed, per-experiment CSVs (`dataset/DDoS/dataset_two_classes.csv`, `dataset/covid/dataset_nuevo_corregido.csv`, `dataset/earthquake/psa_spectral_ratio.csv` + `psa_labels.csv`). `src/tabular_data_to_images/common/dataset.py` reads directly from these paths. Obtain these separately (ask a maintainer, or regenerate the DDoS one — see below) and place them here before running anything.
- **`datasets/bccc-cpacket-cloud-ddos-2024/merged_csv.csv`** (repo root) — the *original, raw* BCCC-cPacket-Cloud-DDoS-2024 export. Only needed if you want to regenerate `dataset/DDoS/...` from scratch via `src/tabular_data_to_images/dataset/processing_scripts/` (`program.py`, `categorical_2.py`, `continous_2.py`). The expected path is `MERGED_CSV_PATH` in `src/tabular_data_to_images/paths.py` — **the repo root is the expected root directory for the original BCCC dataset**, with the raw export placed under `datasets/bccc-cpacket-cloud-ddos-2024/` (sibling to `dataset/`).
- **REFINED mapping pickles** (`src/tabular_data_to_images/refined/{covid,ddos,sismos}/Init_MDS_Euc.pickle`, `Mapping_REFINED.pickle`) — precomputed feature-to-pixel placements. Not committed; either regenerate via the REFINED notebooks + `mpiHill_UF.py` (slow, needs `mpi4py`) or copy them in from wherever they were previously generated.
- **`trained_models/`** (repo root) — training outputs (checkpoints, confusion matrices, result logs) get written here by every script/notebook. Starts empty; populated as you run things.

## What's in here

This repo applies several different **methods** (ways of turning a tabular feature vector into something a classifier can consume) to the same **datasets**. It's easy to conflate the two — e.g. "Sismos" below is a *dataset* (Spanish for "earthquakes/seismic"), not a method.

**Datasets** (3): DDoS network traffic (binary/3-class: Critical / Normal / Suspicious), COVID risk (multi-class risk level), earthquake/PSA soil type (folder name `Sismos`/`sismos`/`psa` interchangeably across the codebase).

**Methods** (each can be pointed at one or more of the datasets above):

| Module | What it does |
|---|---|
| `common/` | Shared plumbing — dataset loading/splitting, training loop helpers, the `SimpleCNN` model. Not a method itself. |
| `baselines/` | Traditional tabular ML directly on the raw feature vectors — **no image transform**: xgboost, logistic regression, random forest, svm, plus dispatch into tabpfn/lstm/transformer. This is the "just use the tabular data as-is" reference point. |
| `lstm/`, `transformer/` | Sequence-model classifiers — treat each tabular feature as one timestep/token in a sequence, fed through an LSTM or Transformer encoder. Still no image transform. |
| `tabpfn/` | [TabPFN](https://github.com/PriorLabs/TabPFN) — a pretrained transformer foundation model for tabular data — plus AutoTabPFN variants, with RFECV/PCA feature-selection front-ends. |
| `deepinsight/` | [DeepInsight](https://github.com/alok-ai-lab/pyDeepInsight): projects features into 2D via t-SNE, arranges them into a fixed image grid per sample, trains a CNN on the resulting images. |
| `refined/` | [REFINED](https://github.com/omidbazgirTTU/REFINED) (vendored, not our code): MDS embedding + a hill-climbing search assigns each *feature* a fixed pixel position (shared across all samples of a dataset); each sample then becomes an image by plugging its values into that fixed layout. |
| `vortex/` | Vortex/VFP (Vortex Feature Positioning): a different feature-to-pixel placement algorithm (correlation-driven), producing images in the same spirit as REFINED but via a different placement search. Self-contained notebooks, not shared/imported code. |
| `tac/` | TAC: turns each sample's feature vector into a convolution kernel, applies it to a fixed base texture image (see `tac/images/`) to produce a per-sample image. Implementation based on the paper linked in `tac/README.md`. |
| `upsampling_approach/` | An original approach (not from a published method): a trainable "vector-to-image" network (transposed convolutions) that learns to turn a feature vector directly into an image, trained jointly with the classifier. |
| `normal/` | The "no image transform, no learned transform" baseline notebook — trains directly on the raw/minimally-processed DDoS data as a reference point separate from `baselines/`. |
| `dataset/processing_scripts/` | One-off scripts that turn the raw BCCC export into the per-`src_ip` aggregated feature CSV used as `dataset/DDoS/...`. Not part of the experiment pipeline itself. |

## Repo layout

```
dataset/                     # NOT committed — processed CSVs (DDoS, covid, earthquake) go here, see "Data"
datasets/                    # NOT committed — raw BCCC dataset goes here, see "Data"
trained_models/              # NOT committed — training outputs, organized per method/dataset
src/tabular_data_to_images/
    common/                   # shared dataset loading, models, training utilities
    baselines/                # xgboost / logistic regression / random forest / svm / tabpfn / lstm / transformer
    lstm/, transformer/       # sequence-model classifiers
    tabpfn/                   # TabPFN + AutoTabPFN variants (RFECV / PCA feature selection)
    deepinsight/              # DeepInsight image-transform pipeline
    tac/                      # TAC image-transform pipeline
    upsampling_approach/      # learned vector-to-image upsampling experiments
    refined/                  # vendored REFINED algorithm + per-dataset notebooks/mappings
    vortex/                   # Vortex/VFP notebooks (self-contained, no importable code)
    normal/                   # "Normal" non-image-transform baseline notebook
    dataset/processing_scripts/  # raw BCCC CSV -> per-src_ip feature aggregation scripts
```

## Running experiments

All entry points are plain Python modules with hardcoded parameters in their `if __name__ == "__main__":` block — edit the script to change dataset/model/hyperparameters, then run via `uv run python -m ...`. There's no CLI/argparse layer.

- **Baselines sweep** (xgboost, lstm, transformer, logistic regression, random forest, svm, deepinsight):
  `uv run python -m tabular_data_to_images.baselines.baselines_train`
- **Single-model baseline + augmentation** (gan / forest_diffusion / llm):
  `uv run python -m tabular_data_to_images.baselines.train`
- **Sequential (LSTM/Transformer) baseline**:
  `uv run python -m tabular_data_to_images.baselines.sequential_train`
- **TabPFN variants**:
  `uv run python -m tabular_data_to_images.tabpfn.rfe_train` (also `rfe_perm_train`, `pca_train`, `pca_hpopt`, `feature_selection`)
- **DeepInsight**:
  `uv run python -m tabular_data_to_images.deepinsight.train`, or open `src/tabular_data_to_images/deepinsight/experiment*.ipynb`
- **TAC**:
  `uv run python -m tabular_data_to_images.tac.train`
- **Upsampling approach**:
  `uv run python -m tabular_data_to_images.upsampling_approach.train` (also `evaluation`, `hyperparam_optimization`)
- **REFINED**: open the notebooks under `src/tabular_data_to_images/refined/{covid,ddos,sismos}/` (e.g. `uv run jupyter lab`). Needs the precomputed mapping pickles in place (see "Data"); the MPI hill-climbing scripts (`mpiHill_UF.py`, etc.) that regenerate them are standalone CLI tools, not imported by the notebooks.
- **Vortex/VFP**: open the notebooks under `src/tabular_data_to_images/vortex/{Covid,DDoS,Sismos}/` — self-contained, no shared code.
- **Normal baseline**: open `src/tabular_data_to_images/normal/DDoS/normal_two_classes.ipynb`.

Training outputs are written under `trained_models/<method>[/<dataset>]/`.
