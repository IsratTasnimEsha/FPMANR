# FPMANR: An Integrated Framework for Stable, Privacy-Preserving, and Attack-Resistant Federated Learning under Non-IID Data

**FedProx + pairwise Masking + Adaptive Noise + Recovery for federated image classification on EMNIST Digits, EMNIST Letters and Fashion-MNIST**

![Python](https://img.shields.io/badge/Python-3-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Implemented-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Paper-in%20preparation-lightgrey)

FPMANR trains a global model across many clients with **heterogeneous (non-IID) data** while protecting the updates that clients send to the server. Each selected client trains locally with **FedProx**, perturbs every layer of its update with **adaptive layer-wise Gaussian noise**, and hides the result with **pairwise masks** shared with its two ring neighbours. The server only sees masked, noisy updates, so the masks cancel in the aggregate. A **client dropout recovery mechanism** keeps aggregation correct when clients fail, without local retraining.

This repository contains the code, results and trained checkpoints for the FPMANR manuscript: training notebooks, gradient-reconstruction and membership-inference attack notebooks, plotting notebooks, result CSVs, figures and checkpoints. It also includes a few **supplementary experiments that are not reported in the manuscript** (see [Supplementary Experiments](#supplementary-experiments-not-in-the-manuscript)).

---

## Table of Contents

- [Overview](#overview)
- [Framework](#framework)
- [Repository Contents](#repository-contents)
- [Supplementary Experiments](#supplementary-experiments-not-in-the-manuscript)
- [Key Results](#key-results)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [Authors and Affiliation](#authors-and-affiliation)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

| Item | Description |
| ---- | ----------- |
| **Task** | Image classification with federated learning under non-IID client data |
| **Datasets** | EMNIST Digits, EMNIST Letters, Fashion-MNIST |
| **Model** | CNN: two 3x3 conv layers (32 and 64 feature maps), each followed by ReLU and 2x2 max-pooling, then a fully connected layer (256 units, ReLU) and an output layer (10 classes for EMNIST Digits and Fashion-MNIST, 26 for EMNIST Letters) |
| **Federation** | 100 clients (10 fixed validation clients, 90 training clients), 30 training clients selected each round, 60 communication rounds |
| **Data partition** | Dirichlet distribution with concentration parameter beta = 0.1 (strongly non-IID) |
| **Methods compared** | FP, FPM, FPMANR 0.1, FPMANR 0.2 (see below) |
| **Attacks evaluated** | Gradient reconstruction (DLG, iDLG, GradInv) and membership inference (confidence, loss, entropy) |

### Method names used in this repository

| Name | Meaning | Adaptive noise factor (alpha) | Notebook suffix in the original code |
| ---- | ------- | ----------------------------- | ------------------------------------ |
| **FP** | FedProx baseline | none | `FedProx` |
| **FPM** | FedProx + pairwise masking | 0.0 | `M_0` |
| **FPMANR 0.1** | FedProx + masking + adaptive noise + recovery | 0.1 | `M_1` |
| **FPMANR 0.2** | FedProx + masking + adaptive noise + recovery | 0.2 | `M_2` |

---

## Framework

```mermaid
flowchart LR
    S["Server: global model"] -->|send to selected clients| C["Client j"]
    C --> T["Local training (FedProx)"]
    T --> N["Adaptive layer-wise Gaussian noise"]
    N --> M["Pairwise ring masking (two neighbours)"]
    M -->|masked noisy update| A["Server aggregation (masks cancel)"]
    A --> S
```

| Component | What it does | Where in the code |
| --------- | ------------ | ----------------- |
| FedProx | Adds a proximal term (mu = 0.01) to the local loss so clients stay close to the global model on non-IID data | `fedprox_loss` in the method notebooks |
| Adaptive noise | For every layer, adds zero-mean Gaussian noise with standard deviation = alpha x norm of that layer's update, so layers with larger updates receive stronger noise | `ADAPTIVE_NOISE_FACTOR` |
| Pairwise masking | Each client exchanges masks with its two ring neighbours, so the masks cancel when the server sums the updates | `MASK_SCALE` (0.001) |
| Dropout recovery | Repairs the masked updates and keeps the aggregate correct when selected clients drop out | `notebooks/client_dropout/` |

Adaptive noise reduces information leakage but does **not** provide a formal differential privacy guarantee.

---

## Repository Contents

### Directory structure

```
FPMANR/
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- requirements.txt
|-- .gitignore
|
|-- notebooks/
|   |-- emnist_digits/            # the same layout is used for emnist_letters/ and fmnist/
|   |   |-- methods/              # 01_fp, 02_fpm, 03_fpmanr_0.1, 04_fpmanr_0.2
|   |   |-- algorithms/           # supplementary: FedAvg / FedSGD versions of FPMANR 0.1 and 0.2
|   |   |-- distributions/        # supplementary: IID and mid-non-IID (beta = 0.5) versions
|   |   |-- attacks/              # gradient_reconstruction, membership_inference
|   |   `-- plots/                # notebooks that draw the comparison graphs
|   |-- emnist_letters/
|   |-- fmnist/
|   `-- client_dropout/           # client_dropout_recovery.ipynb
|
|-- results/
|   |-- emnist_digits/            # the same layout is used for emnist_letters/ and fmnist/
|   |   |-- csv/                  # per-round metrics written by the training notebooks
|   |   |-- figures/              # methods, algorithms, non_iid, attack_reconstruction, attack_membership
|   |   `-- logs/                 # printed output of the attack notebooks
|   |-- emnist_letters/
|   |-- fmnist/
|   `-- client_dropout/           # client_selection_60_rounds.csv, mask_cancellation_figure.png
|
`-- checkpoints/
    |-- emnist_digits/            # fp.pth, fpm.pth, fpmanr_0.1.pth, fpmanr_0.2.pth
    |-- emnist_letters/
    `-- fmnist/
```

### File and directory reference

| Path | Description | In the manuscript? |
| ---- | ----------- | ------------------ |
| `notebooks/{emnist_digits,emnist_letters,fmnist}/methods/` | Federated training for FP, FPM, FPMANR 0.1 and FPMANR 0.2 (non-IID, beta = 0.1) | Yes |
| `notebooks/{emnist_digits,emnist_letters,fmnist}/attacks/` | Gradient reconstruction (DLG, iDLG, GradInv) and membership inference against the saved checkpoints | Yes |
| `notebooks/{emnist_digits,emnist_letters,fmnist}/plots/` | Plotting notebooks for the method graphs and the dropout figure | Yes (method and dropout plots) |
| `notebooks/client_dropout/` | Client dropout and mask-recovery experiment | Yes |
| `notebooks/{emnist_digits,emnist_letters,fmnist}/algorithms/` | FPMANR trained with FedAvg and FedSGD aggregation | Supplementary |
| `notebooks/{emnist_digits,emnist_letters,fmnist}/distributions/` | FPMANR trained on IID and beta = 0.5 partitions | Supplementary |
| `results/*/csv/` | Round-by-round accuracy, loss and time for each run | Yes |
| `results/*/figures/` | Figures produced from the CSVs and attack notebooks | Yes, except `algorithms/` and `non_iid/` |
| `results/*/logs/` | Text output of the attack notebooks (metrics printed during the runs) | Yes |
| `checkpoints/*/` | Trained global models (about 3 MB each) | Yes |

### Which notebook writes which file

| Notebook | CSV written (`results/*/csv/`) | Checkpoint |
| -------- | -------------------------------------- | ---------- |
| `methods/01_fp` | `{EMNIST_Digits,EMNIST_Letters,FMNIST}_fedprox_federated_metrics_0.csv` | `fp.pth` |
| `methods/02_fpm` | `{EMNIST_Digits,EMNIST_Letters,FMNIST}_masking_federated_metrics_0.csv` | `fpm.pth` |
| `methods/03_fpmanr_0.1` | `{EMNIST_Digits,EMNIST_Letters,FMNIST}_masking_federated_metrics_1.csv` | `fpmanr_0.1.pth` |
| `methods/04_fpmanr_0.2` | `{EMNIST_Digits,EMNIST_Letters,FMNIST}_masking_federated_metrics_2.csv` | `fpmanr_0.2.pth` |
| `algorithms/fpmanr_0.1_fedavg` / `_fedsgd` (supplementary) | `{EMNIST_Digits,EMNIST_Letters,FMNIST}_masking_federated_metrics_1_fedavg.csv` / `_fedsgd.csv` | |
| `algorithms/fpmanr_0.2_fedavg` / `_fedsgd` (supplementary) | `{EMNIST_Digits,EMNIST_Letters,FMNIST}_masking_federated_metrics_2_fedavg.csv` / `_fedsgd.csv` | |
| `distributions/fpmanr_0.1_iid` / `_mid` (supplementary) | `IID_{EMNIST_Digits,EMNIST_Letters,FMNIST}_masking_federated_metrics_1.csv` / `Mid_{EMNIST_Digits,EMNIST_Letters,FMNIST}_masking_federated_metrics_1.csv` | |
| `distributions/fpmanr_0.2_iid` / `_mid` (supplementary) | `IID_{EMNIST_Digits,EMNIST_Letters,FMNIST}_masking_federated_metrics_2.csv` / `Mid_{EMNIST_Digits,EMNIST_Letters,FMNIST}_masking_federated_metrics_2.csv` | |

The file-name prefix is `EMNIST_Digits`, `EMNIST_Letters` or `FMNIST`, depending on the dataset.

---

## Supplementary Experiments (not in the manuscript)

The following experiments were run while preparing the paper. They are provided for completeness and reproducibility, but **they are not reported in the manuscript and are not part of its claims**:

| Experiment | Location | What it does |
| ---------- | -------- | ------------ |
| Aggregation algorithms | `notebooks/*/algorithms/`, `results/*/figures/algorithms/` | FPMANR 0.1 and 0.2 trained with FedAvg and FedSGD aggregation |
| Data distributions | `notebooks/*/distributions/`, `results/*/figures/non_iid/` | FPMANR 0.1 and 0.2 trained on IID and mid-non-IID (beta = 0.5) partitions |

Everything else in this repository (methods, attacks, client dropout, plots of the main comparison) corresponds to the experiments reported in the manuscript.

---

## Key Results

Final-round (round 60) test accuracy (%), non-IID with beta = 0.1, as reported in the manuscript and computed from the CSV files in `results/*/csv/`:

| Method | EMNIST Digits | EMNIST Letters | Fashion-MNIST |
| ------ | ------------- | -------------- | ------------- |
| FP | 97.92 | 90.53 | 80.57 |
| FPM | 98.52 | 90.45 | 73.71 |
| FPMANR 0.1 | 98.10 | 85.39 | 74.51 |
| FPMANR 0.2 | 96.57 | 73.37 | 73.12 |

Values are rounded to two decimals. Where the stored CSV value lies exactly on a rounding boundary (for example 98.105 for FPMANR 0.1 on EMNIST Digits), the manuscript's rounding is used.

- Adding adaptive noise follows the expected privacy-utility trade-off: FPMANR 0.1 keeps high accuracy, while FPMANR 0.2 reduces it further, most visibly on EMNIST Letters.
- FPMANR 0.1 gives the best balance between privacy and utility among the evaluated settings.
- FP and FPM would be identical in theory because the pairwise masks cancel during aggregation. The small differences, larger on Fashion-MNIST, come from floating-point rounding errors in the masked aggregation.
- Adaptive noise substantially reduces the success of gradient reconstruction attacks (DLG, iDLG, GradInv): MSE and LPIPS increase while PSNR, SSIM and reconstruction success rate decrease.
- Membership inference attacks stay close to random guessing, with AUC values around 0.5 and attack accuracy near 50%.
- Full curves, attack figures and printed attack metrics are in `results/*/figures/` and `results/*/logs/`.

---

## Datasets

| Dataset | Source | How it is obtained |
| ------- | ------ | ------------------ |
| EMNIST Digits | `torchvision.datasets.EMNIST(split="digits")` | Downloaded automatically into `./data` |
| EMNIST Letters | `torchvision.datasets.EMNIST(split="letters")` | Downloaded automatically into `./data` |
| Fashion-MNIST | `torchvision.datasets.FashionMNIST` | Downloaded automatically into `./data` |

Data handling: EMNIST images are normalised with mean 0.1307 and standard deviation 0.3081, and Fashion-MNIST images with mean 0.5 and standard deviation 0.5. The training set is split across 100 clients with a Dirichlet distribution (beta = 0.1). Clients 0 to 9 are held out as validation clients and clients 10 to 99 are training clients. Accuracy is reported on the official test set. The `data/` folder is not stored in this repository.

---

## Installation

```bash
git clone https://github.com/IsratTasnimEsha/FPMANR.git
cd FPMANR

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

| Package | Purpose |
| ------- | ------- |
| `torch`, `torchvision` | Models, training and dataset download |
| `numpy`, `pandas` | Numerics and CSV handling |
| `matplotlib` | Figures |
| `scikit-learn` | ROC curves and AUC for membership inference |
| `scikit-image`, `lpips` | SSIM and LPIPS for reconstruction quality |

---

## Usage

The experiments are Jupyter notebooks written for **Google Colab**. Some cells call `google.colab.files.download`; when running locally, remove or skip those cells. Each notebook is self-contained: it downloads the dataset, trains, and writes its CSV and checkpoint to the working directory.

1. **Train the four methods** for a dataset, in the order given by the file names:
   `notebooks/emnist_digits/methods/01_fp.ipynb` to `04_fpmanr_0.2.ipynb`.
2. **Move outputs into place:** put each CSV file in the `csv/` folder of its dataset under `results/`, and each `.pth` file in the folder of its dataset under `checkpoints/`.
3. **Attacks:** run `attacks/gradient_reconstruction.ipynb` and `attacks/membership_inference.ipynb` against the saved checkpoints.
4. **Figures:** run the notebooks in `plots/` to redraw the comparison graphs from the CSV files.
5. **Client dropout:** run `notebooks/client_dropout/client_dropout_recovery.ipynb`.
6. **Supplementary (optional, not in the manuscript):** run the notebooks in `algorithms/` (FedAvg, FedSGD) and `distributions/` (IID, beta = 0.5).

All notebooks are stored without outputs to keep the repository small. The results they produced are in `results/`.

---

## Hyperparameters

| Hyperparameter | Value |
| -------------- | ----- |
| Number of clients | 100 (10 validation, 90 training) |
| Clients selected per round | 30 (30% of all clients, chosen from the 90 training clients) |
| Communication rounds | 60 |
| Local epochs | 3 |
| Batch size | 128 |
| Local optimizer | SGD, learning rate 0.01, momentum 0.9, weight decay 5e-4 |
| FedProx coefficient (mu) | 0.01 |
| Dirichlet concentration parameter (beta) | 0.1 (main experiments); 0.5 and IID in the supplementary experiments |
| Adaptive noise factor (alpha) | 0.0 (FPM), 0.1 (FPMANR 0.1), 0.2 (FPMANR 0.2) |
| Mask scale | 0.001 |
| Random seed | 42 |

---

## Authors and Affiliation

**Israt Tasnim Esha<sup>a,\*</sup>, Dola Das<sup>b</sup>, Nusrat Jahan Jonaki<sup>c</sup>**

<sup>a</sup> Department of Computer Science and Engineering (CSE), University of Brahmanbaria (UoB), Datiara, Brahmanbaria-3400, Bangladesh.

<sup>b</sup> Department of Computer Science and Engineering (CSE), Khulna University of Engineering & Technology (KUET), Khulna-9203, Bangladesh.

<sup>c</sup> Department of Computer Science and Engineering (CSE), Southeast University (SEU), Tejgaon, Dhaka-1208, Bangladesh.

<sup>\*</sup> Corresponding author: israttasnimesha1046@gmail.com

---

## Citation

The FPMANR journal manuscript is in preparation. Until it is available, please cite the conference paper this work extends:

```bibtex
@inproceedings{esha2026fpman,
  title     = {FPMAN: A Framework for Stable and Privacy-Preserving Federated Learning under Non-IID Data},
  author    = {Esha, Israt Tasnim and Tasnim, Tasfia and Das, Dola},
  booktitle = {2026 International Conference on Power, Electronics, Communications, Computing, and Intelligent Infrastructure (PECCII)},
  pages     = {1--6},
  year      = {2026},
  organization = {IEEE},
  doi       = {10.1109/PECCII70991.2026.11662022}
}
```

---

## License

The code is released under the MIT License (see `LICENSE`).

---

## Contact

For questions about the code or paper, please open a GitHub Issue or write to Israt Tasnim Esha: israttasnimesha1046@gmail.com
