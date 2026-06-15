# Spatial Communication Graph Learning Workflow

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" />
  <img src="https://img.shields.io/badge/Status-Research%20Code-lightgrey" />
  <img src="https://img.shields.io/badge/Domain-Spatial%20Transcriptomics-green" />
</p>

<p align="center">
  <b>A research workflow for spatial cell-cell communication graph construction and heterogeneous graph learning.</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#pipeline">Pipeline</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

This repository contains research code for constructing spatial cell-cell communication graphs from single-cell and spatial transcriptomics data.

The workflow integrates:

* snRNA-seq or scRNA-seq cell-level features
* Visium spatial transcriptomics spot-level features
* Tangram cell-to-spot mapping
* Ligand-receptor feature matrices
* CellChat-derived significant interaction pairs
* Spatial neighbor graphs
* Structure-aware heterogeneous graph model training

The code is designed for paper reproduction, method development, and downstream spatial communication analysis.

---

## Pipeline

```text
Input data
  |
  |-- snRNA-seq / scRNA-seq AnnData
  |-- Visium spatial AnnData
  |-- Tangram cell-to-spot mapping
  |-- Ligand-receptor feature matrix
  |-- CellChat significant interaction pairs
  |
  v
Feature construction
  |
  |-- Cell features
  |-- Spot features
  |-- Ligand-receptor node features
  |-- Pathway / TF activity features
  |
  v
Graph construction
  |
  |-- Sender edges
  |-- Receiver edges
  |-- Spot-present LR edges
  |-- Spatial neighbor edges
  |-- Spot structural edges
  |
  v
Model training
  |
  |-- Structure-biased heterogeneous graph model
  |-- Auxiliary pathway / TF objectives
  |
  v
Outputs
  |-- Graph files
  |-- Learned embeddings
  |-- Model checkpoints
  |-- Downstream communication scores
```

---

## Repository Structure

```text
paper-code/
├── build_cell_at_spot_geomprob_from_tangram.py
├── build_cell_features_final_from_existing_cli.py
├── build_receiver_edges.py
├── build_sender_edges.py
├── build_spot_features_and_neighbors.py
├── build_spot_present_lr.spatial.py
├── build_spot_spstruct_edges.py
├── cellchat_lr_filter_edges.py
├── decoupler_waggr_to_spot_cli.py
├── loop_train_score_spatial.py
├── lr_build_node_features.py
├── mirror_spot_lr_edges.py
├── prune_and_filter_tangram_map_cli.py
├── train_routeA_structbias.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/dongwu2/paper-code.git
cd paper-code
```

Create a clean Python environment:

```bash
conda create -n spatial-graph python=3.9 -y
conda activate spatial-graph
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> Note: PyTorch and PyTorch Geometric installation may depend on your CUDA version. If installation fails, please install the versions matching your local CUDA / CPU environment.

---

## Required Dependencies

The main dependencies include:

```text
numpy
pandas
scipy
scikit-learn
anndata
scanpy
h5py
pyyaml
tqdm
networkx
matplotlib
seaborn
torch
torch-geometric
decoupler
```

If a Snakemake workflow is used, install:

```text
snakemake
```

---

## Input Data

| Input                    | Format  | Description                                          |
| ------------------------ | ------- | ---------------------------------------------------- |
| Single-cell data         | `.h5ad` | snRNA-seq or scRNA-seq AnnData object                |
| Spatial data             | `.h5ad` | Visium AnnData object with spatial coordinates       |
| Tangram mapping          | `.h5ad` | Cell-to-spot mapping result                          |
| Ligand-receptor features | `.h5ad` | Ligand-receptor feature matrix                       |
| CellChat pairs           | `.csv`  | Significant sender and receiver LR interaction pairs |

Large data files are not included in this repository. Please prepare the required `.h5ad` and `.csv` files before running the scripts.

---

## Main Scripts

| Script                                           | Description                                                            |
| ------------------------------------------------ | ---------------------------------------------------------------------- |
| `prune_and_filter_tangram_map_cli.py`            | Prune and filter Tangram cell-to-spot mapping                          |
| `build_cell_features_final_from_existing_cli.py` | Build cell-level features                                              |
| `build_spot_features_and_neighbors.py`           | Build spatial spot features and spot-neighbor graph                    |
| `build_cell_at_spot_geomprob_from_tangram.py`    | Build cell-at-spot probability features from Tangram mapping           |
| `lr_build_node_features.py`                      | Build ligand-receptor node features                                    |
| `decoupler_waggr_to_spot_cli.py`                 | Aggregate pathway and transcription factor activities to spatial spots |
| `build_sender_edges.py`                          | Construct sender-side communication edges                              |
| `build_receiver_edges.py`                        | Construct receiver-side communication edges                            |
| `cellchat_lr_filter_edges.py`                    | Filter ligand-receptor edges using CellChat significant pairs          |
| `build_spot_present_lr.spatial.py`               | Build spot-present ligand-receptor edges                               |
| `build_spot_spstruct_edges.py`                   | Build spatial structural edges between spots                           |
| `mirror_spot_lr_edges.py`                        | Mirror spot-ligand-receptor edges when needed                          |
| `train_routeA_structbias.py`                     | Train the structure-biased heterogeneous graph model                   |
| `loop_train_score_spatial.py`                    | Run iterative training and scoring                                     |

---

## Usage

The scripts can be run individually according to the pipeline order.

A typical workflow contains the following stages:

### 1. Prune Tangram mapping

```bash
python prune_and_filter_tangram_map_cli.py \
  --input /path/to/tangram_map.h5ad \
  --output /path/to/pruned_tangram_map.h5ad
```

### 2. Build cell features

```bash
python build_cell_features_final_from_existing_cli.py \
  --input /path/to/sc_data.h5ad \
  --output /path/to/cell_features.h5ad
```

### 3. Build spot features and spatial neighbors

```bash
python build_spot_features_and_neighbors.py \
  --input /path/to/spatial_data.h5ad \
  --output /path/to/spot_features.h5ad
```

### 4. Build ligand-receptor node features

```bash
python lr_build_node_features.py \
  --input /path/to/lr_features.h5ad \
  --output /path/to/lr_node_features.h5ad
```

### 5. Build communication graph edges

```bash
python build_sender_edges.py \
  --input /path/to/input_files \
  --output /path/to/sender_edges.csv

python build_receiver_edges.py \
  --input /path/to/input_files \
  --output /path/to/receiver_edges.csv
```

### 6. Filter edges using CellChat results

```bash
python cellchat_lr_filter_edges.py \
  --sender_pairs /path/to/sender_sig_pairs.csv \
  --receiver_pairs /path/to/receiver_sig_pairs.csv \
  --output /path/to/filtered_edges.csv
```

### 7. Train the graph model

```bash
python train_routeA_structbias.py \
  --epochs_pre 5 \
  --epochs_ft 20 \
  --amp \
  --batch_size 256 \
  --lambda_aux 0.1 \
  --lambda_aux_path 0.1 \
  --lambda_aux_tf 0.1
```

### 8. Run iterative training and scoring

```bash
python loop_train_score_spatial.py \
  --rounds 3
```

> The exact command-line arguments may vary depending on the local data layout and script version. Please check each script for available arguments.

---

## Configuration

The workflow can be controlled by a YAML configuration file.

Example:

```yaml
rounds: 3

base_root: "/path/to/output/root"

defaults:
  prune_tangram_map:
    topk_per_row: 50
    cumm_mass: 0.85
    p_min: 0.01
    severity: "mild"

  cell_features:
    celltype_key: "cell_type"
    n_pcs: 50

  spot_features:
    k_min_neighbors: 6
    pe_dim: 16

  cell_at_spot:
    min_prob: 0.01
    topm_per_cell: 3

  spot_present_lr:
    mode: "knn"
    knn_k: 6
    smooth_k: 2
    smooth_alpha: 0.6
    ligand_agg: "min"
    only_triad_lr: true

  train:
    train_extra_args: >-
      --epochs_pre 5
      --epochs_ft 20
      --amp
      --batch_size 256
      --lambda_aux 0.1
      --lambda_aux_path 0.1
      --lambda_aux_tf 0.1

samples:
  Sample_01:
    tangram_ad_map_raw: "ad_map_best.h5ad"
    sc_h5ad: "/path/to/snRNA_or_scRNA.h5ad"
    st_h5ad: "/path/to/spatial.with_spatial.rawX.h5ad"
    lr_h5ad: "/path/to/lr_features.h5ad"
    sender_sig_pairs_csv: "/path/to/sender_sig_pairs.csv"
    receiver_sig_pairs_csv: "/path/to/receiver_sig_pairs.csv"
    cellchat_thresh: 0.05
    train_script: "/path/to/train_routeA_structbias.py"
```

Please replace all local paths with paths on your own machine before running the workflow.

---

## Optional Snakemake Usage

If a `Snakefile` and configuration file are available, the workflow can be executed with Snakemake:

```bash
snakemake --configfile configs/example_config.yaml --cores 8
```

For a dry run:

```bash
snakemake --configfile configs/example_config.yaml --cores 8 -n
```

If no `Snakefile` is provided, please run the scripts individually according to the pipeline order.

---

## Outputs

The workflow produces intermediate graph files, model checkpoints, learned embeddings, and downstream communication scores under the configured output directory.

Example output structure:

```text
/path/to/output/root/Sample_01/
├── features/
├── edges/
├── checkpoints/
├── embeddings/
└── scores/
```

---

## Notes

* This repository contains research code and may require path adjustments for different datasets.
* Large raw data files are not included.
* Please verify all input paths before running the workflow.
* Some dependencies may require specific versions depending on your system environment.

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{yourpaper2026,
  title   = {Your Paper Title},
  author  = {Author One and Author Two},
  journal = {Journal or Conference},
  year    = {2026},
  url     = {https://github.com/dongwu2/paper-code}
}
```

---

## Contact

For questions, please open an issue or contact the repository maintainer.

---

## License

Please add license information before public release.
