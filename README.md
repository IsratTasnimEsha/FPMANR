# FPMANR: Stable, Privacy-Preserving, and Attack-Resistant Federated Learning under Non-IID Data

**FedProx + pairwise Masking + Adaptive Noise + Recovery for federated image classification on EMNIST Digits, EMNIST Letters and Fashion-MNIST**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Implemented-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Paper-in%20preparation-lightgrey)

FPMANR trains a global model across many clients with **heterogeneous (non-IID) data** while protecting the updates that clients send to the server. Each selected client trains locally with **FedProx**, perturbs every layer of its update with **adaptive Gaussian noise**, and hides the result with **pairwise masks** shared with its ring neighbours. The server only sees masked, noisy updates, so the masks cancel in the aggregate. A **recovery mechanism** keeps aggregation correct when clients drop out. The repository contains the training notebooks, gradient-reconstruction and membership-inference attack notebooks, plotting notebooks, result CSVs, figures and trained checkpoints behind the paper.

---

## Table of Contents

- [Overview](#overview)
- [Framework](#framework)
- [Repository Contents](#repository-contents)
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
| **Model** | Small CNN: two 3x3 conv layers (32, 64 channels) with max-pooling, then fully connected layers (256 units, then one unit per class) |
| **Federation** | 100 clients, 30% selected each round, 60 communication rounds |
| **Data partition** | Dirichlet with alpha = 0.1 (strongly non-IID); alpha = 0.5 and IID for comparison |
| **Methods compared** | FP, FPM, FPMANR 0.1, FPMANR 0.2 (see below) |
| **Aggregation algorithms compared** | FedAvg, FedSGD, FedProx |
| **Attacks evaluated** | Gradient reconstruction (DLG, iDLG, GradInv) and membership inference (confidence, loss, entropy) |

### Method names used in this repository

| Name | Meaning | Adaptive noise factor | Notebook suffix in the original code |
| ---- | ------- | --------------------- | ------------------------------------ |
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
    N --> M["Pairwise ring masking"]
    M -->|masked noisy update| A["Server aggregation (masks cancel)"]
    A --> S
```

| Component | What it does | Where in the code |
| --------- | ------------ | ----------------- |
| FedProx | Adds a proximal term (mu = 0.01) to the local loss so clients stay close to the global model on non-IID data | `fedprox_loss` in the method notebooks |
| Adaptive noise | For every layer, adds Gaussian noise with standard deviation = noise factor x norm of that layer's update | `ADAPTIVE_NOISE_FACTOR` |
| Pairwise masking | Each client adds a random mask and subtracts the same mask from its ring neighbour, so masks cancel when the server sums the updates | `MASK_SCALE` (0.001) |
| Dropout recovery | Keeps the aggregate correct when selected clients drop out | `notebooks/client_dropout/` |

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
|   |   |-- algorithms/           # FedAvg / FedSGD versions of FPMANR 0.1 and 0.2
|   |   |-- distributions/        # IID and mid-non-IID (alpha = 0.5) versions
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

| Path | Description |
| ---- | ----------- |
| `notebooks/<dataset>/methods/` | Federated training for FP, FPM, FPMANR 0.1 and FPMANR 0.2 (non-IID, alpha = 0.1) |
| `notebooks/<dataset>/algorithms/` | FPMANR trained with FedAvg and FedSGD aggregation, for the algorithm comparison |
| `notebooks/<dataset>/distributions/` | FPMANR trained on IID and alpha = 0.5 partitions, for the non-IID comparison |
| `notebooks/<dataset>/attacks/` | Gradient reconstruction (DLG, iDLG, GradInv) and membership inference against the saved checkpoints |
| `notebooks/<dataset>/plots/` | Plotting notebooks for the method, algorithm, non-IID and dropout graphs |
| `notebooks/client_dropout/` | Client dropout and mask-recovery experiment |
| `results/<dataset>/csv/` | Round-by-round accuracy, loss and time for each run |
| `results/<dataset>/figures/` | Figures produced from the CSVs and attack notebooks |
| `results/<dataset>/logs/` | Text output of the attack notebooks (metrics printed during the runs) |
| `checkpoints/<dataset>/` | Trained global models (about 3 MB each) |

### Which notebook writes which file

| Notebook | CSV written (`results/<dataset>/csv/`) | Checkpoint |
| -------- | -------------------------------------- | ---------- |
| `methods/01_fp` | `<DATASET>_fedprox_federated_metrics_0.csv` | `fp.pth` |
| `methods/02_fpm` | `<DATASET>_masking_federated_metrics_0.csv` | `fpm.pth` |
| `methods/03_fpmanr_0.1` | `<DATASET>_masking_federated_metrics_1.csv` | `fpmanr_0.1.pth` |
| `methods/04_fpmanr_0.2` | `<DATASET>_masking_federated_metrics_2.csv` | `fpmanr_0.2.pth` |
| `algorithms/fpmanr_0.1_fedavg` / `_fedsgd` | `<DATASET>_masking_federated_metrics_1_fedavg.csv` / `_fedsgd.csv` | |
| `algorithms/fpmanr_0.2_fedavg` / `_fedsgd` | `<DATASET>_masking_federated_metrics_2_fedavg.csv` / `_fedsgd.csv` | |
| `distributions/fpmanr_0.1_iid` / `_mid` | `IID_<DATASET>_masking_federated_metrics_1.csv` / `Mid_<DATASET>_masking_federated_metrics_1.csv` | |
| `distributions/fpmanr_0.2_iid` / `_mid` | `IID_<DATASET>_masking_federated_metrics_2.csv` / `Mid_<DATASET>_masking_federated_metrics_2.csv` | |

`<DATASET>` is `EMNIST_Digits`, `EMNIST_Letters` or `FMNIST`.

---

## Key Results

Final-round test accuracy (%) after 60 rounds, non-IID with alpha = 0.1, computed from the CSV files in `results/<dataset>/csv/`:

| Method | EMNIST Digits | EMNIST Letters | Fashion-MNIST |
| ------ | ------------- | -------------- | ------------- |
| FP | 97.92 | 90.53 | 80.57 |
| FPM | 98.52 | 90.45 | 73.71 |
| FPMANR 0.1 | 98.11 | 85.39 | 74.51 |
| FPMANR 0.2 | 96.58 | 73.38 | 73.12 |

- On **EMNIST Digits** every method stays above 96%, and FPMANR 0.1 loses almost nothing compared with FP.
- Adding more noise costs accuracy: FPMANR 0.2 is clearly lower than FPMANR 0.1 on EMNIST Letters.
- Fashion-MNIST accuracy fluctuates from round to round under alpha = 0.1, so the last-round value is lower than the best round. The full curves are in `results/fmnist/figures/methods/`.
- Attack results (reconstruction quality and membership-inference AUC) are in `results/<dataset>/figures/` and `results/<dataset>/logs/`.

---

## Datasets

| Dataset | Source | How it is obtained |
| ------- | ------ | ------------------ |
| EMNIST Digits | `torchvision.datasets.EMNIST(split="digits")` | Downloaded automatically into `./data` |
| EMNIST Letters | `torchvision.datasets.EMNIST(split="letters")` | Downloaded automatically into `./data` |
| Fashion-MNIST | `torchvision.datasets.FashionMNIST` | Downloaded automatically into `./data` |

Data handling: EMNIST images are normalised with mean 0.1307 and standard deviation 0.3081, and Fashion-MNIST images with mean 0.5 and standard deviation 0.5. The training set is split across 100 clients with a Dirichlet distribution. Clients 0 to 9 are held out as validation clients and clients 10 to 99 are training clients. Accuracy is reported on the official test set. The `data/` folder is not stored in this repository.

---

## Installation

```bash
git clone https://github.com/IsratTasnimEsha/<repository-name>.git
cd <repository-name>

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
2. **Optional comparisons:** run the notebooks in `algorithms/` (FedAvg, FedSGD) and `distributions/` (IID, alpha = 0.5).
3. **Move outputs into place:** put the CSV files in `results/<dataset>/csv/` and the `.pth` files in `checkpoints/<dataset>/`.
4. **Attacks:** run `attacks/gradient_reconstruction.ipynb` and `attacks/membership_inference.ipynb` against the saved checkpoints.
5. **Figures:** run the notebooks in `plots/` to redraw the comparison graphs from the CSV files.
6. **Client dropout:** run `notebooks/client_dropout/client_dropout_recovery.ipynb`.

All notebooks are stored without outputs to keep the repository small. The results they produced are in `results/`.

---

## Hyperparameters

| Hyperparameter | Value |
| -------------- | ----- |
| Number of clients | 100 (10 validation, 90 training) |
| Clients selected per round | 30 (30%) |
| Communication rounds | 60 |
| Local epochs | 3 |
| Batch size | 128 |
| Local optimizer | SGD, learning rate 0.01, momentum 0.9, weight decay 5e-4 |
| FedProx coefficient (mu) | 0.01 |
| Dirichlet alpha | 0.1 (main), 0.5 (mid non-IID), IID |
| Adaptive noise factor | 0.0 (FPM), 0.1 (FPMANR 0.1), 0.2 (FPMANR 0.2) |
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

GitHub also shows a "Cite this repository" button, which reads `CITATION.cff`.

---

## License

The code is released under the MIT License (see `LICENSE`).

---

## Contact

For questions about the code or paper, please open a GitHub Issue or write to Israt Tasnim Esha: israttasnimesha1046@gmail.com
