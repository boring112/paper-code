# Spatial Communication Graph Learning Workflow

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" />
  <img src="https://img.shields.io/badge/Workflow-Snakemake-green" />
  <img src="https://img.shields.io/badge/Status-Research%20Code-lightgrey" />
</p>

<p align="center">
  <b>A workflow for spatial cell–cell communication graph construction and heterogeneous graph learning.</b>
</p>

---

## Overview

This repository contains research code for constructing spatial cell–cell communication graphs from single-cell and spatial transcriptomics data.

The workflow integrates:

* snRNA-seq cell-level features
* Visium spatial transcriptomics spot-level features
* Tangram cell-to-spot mapping
* Ligand–receptor feature matrices
* CellChat-derived significant interaction pairs
* Spatial neighbor graphs
* Structure-biased heterogeneous graph model training

This code is designed for paper reproduction and downstream spatial communication analysis.

---

## Pipeline

```text
Input data
  |
  |-- snRNA-seq AnnData
  |-- Visium spatial AnnData
  |-- Tangram cell-to-spot mapping
  |-- Ligand-receptor features
  |-- CellChat significant LR pairs
  |
  v
Feature construction
  |
  |-- Cell features
  |-- Spot features
  |-- LR features
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
  v
Outputs
  |-- Learned embeddings
  |-- Training checkpoints
  |-- Downstream scores
```

---

## Repository Structure

```text
paper-code/
├── build_cell_features_final_from_existing_cli.py
├── build_spot_features_and_neighbors.py
├── build_sender_edges.py
├── build_receiver_edges.py
├── build_spot_present_lr.spatial.py
├── build_spot_spstruct_edges.py
├── build_cell_at_spot_geomprob_from_tangram.py
├── prune_and_filter_tangram_map_cli.py
├── cellchat_lr_filter_edges.py
├── mirror_spot_lr_edges.py
├── train_routeA_structbias.py
├── loop_train_score_spatial.py
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

Create a Python environment:

```bash
conda create -n spatial-graph python=3.9 -y
conda activate spatial-graph
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

The workflow is controlled by a YAML configuration file.

Example:

```yaml
rounds: 3

base_root: "/path/to/output/root"

defaults:
  prune_tangram_map:
    topk_per_row: 50
    cumm_mass: 0.85
    p_min: 0.01

  cell_features:
    celltype_key: "cell_type"
    n_pcs: 50

  spot_features:
    k_min_neighbors: 6
    pe_dim: 16

  train:
    train_extra_args: >-
      --epochs_pre 5
      --epochs_ft 20
      --amp
      --batch_size 256
      --lambda_aux 0.1
      --lambda_aux_path 0.1
      --lambda_aux_tf 0.1
```

Each sample can be specified under the `samples` section:

```yaml
samples:
  Patient_27:
    tangram_ad_map_raw: "ad_map_best.h5ad"
    sc_h5ad: "/path/to/snRNA.h5ad"
    st_h5ad: "/path/to/spatial.with_spatial.rawX.h5ad"
    lr_h5ad: "/path/to/lr_features.h5ad"
    sender_sig_pairs_csv: "/path/to/sender_sig_pairs.csv"
    receiver_sig_pairs_csv: "/path/to/receiver_sig_pairs.csv"
    cellchat_thresh: 0.05
    train_script: "/path/to/train_routeA_structbias.py"
```

Please replace all local paths with paths on your own machine before running the workflow.

---

## Usage

If using Snakemake:

```bash
snakemake --configfile configs/27_config.yaml --cores 8
```

For a dry run:

```bash
snakemake --configfile configs/27_config.yaml --cores 8 -n
```

The model training script can also be run manually:

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

---

## Main Scripts

| Script                                           | Description                                          |
| ------------------------------------------------ | ---------------------------------------------------- |
| `prune_and_filter_tangram_map_cli.py`            | Prune and filter Tangram cell-to-spot mapping        |
| `build_cell_features_final_from_existing_cli.py` | Build cell-level features                            |
| `build_spot_features_and_neighbors.py`           | Build spatial spot features and neighbor graph       |
| `build_cell_at_spot_geomprob_from_tangram.py`    | Build cell-at-spot probability features from Tangram |
| `build_sender_edges.py`                          | Construct sender-side communication edges            |
| `build_receiver_edges.py`                        | Construct receiver-side communication edges          |
| `cellchat_lr_filter_edges.py`                    | Filter LR edges using CellChat significant pairs     |
| `build_spot_present_lr.spatial.py`               | Build spot-present ligand-receptor edges             |
| `build_spot_spstruct_edges.py`                   | Build spatial structural edges between spots         |
| `mirror_spot_lr_edges.py`                        | Mirror spot-LR edges when needed                     |
| `train_routeA_structbias.py`                     | Train the structure-biased heterogeneous graph model |
| `loop_train_score_spatial.py`                    | Run iterative training and scoring                   |

---

## Inputs

| Input                        | Format  | Description                                          |
| ---------------------------- | ------- | ---------------------------------------------------- |
| snRNA-seq data               | `.h5ad` | Single-cell or single-nucleus AnnData                |
| Spatial transcriptomics data | `.h5ad` | Visium AnnData with spatial coordinates              |
| Tangram map                  | `.h5ad` | Cell-to-spot mapping result                          |
| LR features                  | `.h5ad` | Ligand-receptor feature matrix                       |
| CellChat pairs               | `.csv`  | Significant sender and receiver LR interaction pairs |

---

## Outputs

The workflow produces graph files, model checkpoints, learned embeddings, and downstream scores under the configured output directory:

```text
{base_root}/{sample}/
```

Example:

```text
/path/to/output/root/Patient_27/
```

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
