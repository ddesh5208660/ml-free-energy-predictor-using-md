# ML-Assisted Free Energy Landscape Prediction from Molecular Dynamics and Metadynamics

> Predicting the free energy landscape of siRNA–NAPA bio-oligomer complexes using machine learning trained on all-atom MD and well-tempered metadynamics trajectories.

---

## Overview

Understanding how siRNA binds to polymer delivery vehicles requires mapping a high-dimensional free energy surface — a task that is computationally expensive even with enhanced sampling. This project builds a machine learning surrogate model that learns to predict the free energy landscape directly from structural and dynamical features extracted from MD trajectories and well-tempered metadynamics simulations.

The trained model reproduces the metadynamics free energy surface (FES) at a fraction of the computational cost, enabling rapid screening of collective variable (CV) space without running new simulations.

---

## Key Features

- Extracts structural descriptors (distances, angles, contacts, RMSD, Rg) from GROMACS `.xtc` trajectories using MDAnalysis
- Parses HILLS files and `COLVAR` output from PLUMED to reconstruct the metadynamics free energy surface
- Trains gradient boosting and neural network regressors (scikit-learn, PyTorch) on CV → ΔG mappings
- Evaluates model accuracy against plumed's `sum_hills` reference FES
- Produces publication-quality free energy contour plots and parity plots

---

## Scientific Context

**System:** siRNA–NAPA (N-acryloyl amino acid) bio-oligomer complexes in explicit solvent  
**Force field:** CHARMM36m (nucleic acids) + custom polymer parameters  
**Simulation engine:** GROMACS 2023 + PLUMED 2.9  
**Enhanced sampling:** Well-tempered metadynamics (WT-MetaD) with bias factor γ = 15  
**Collective variables:** End-to-end distance, radius of gyration, number of contacts  
**ML objective:** Predict ΔG(CV₁, CV₂) from structural features without running WT-MetaD

---

## Repository Structure

```
.
├── data/
│   ├── trajectories/          # GROMACS .xtc files (not tracked — see Data section)
│   ├── hills/                 # PLUMED HILLS files from metadynamics runs
│   ├── colvar/                # COLVAR output for each replica
│   └── fes_reference/         # Reference FES from sum_hills (2D grids, .dat)
│
├── features/
│   ├── extract_features.py    # MDAnalysis-based descriptor extraction
│   ├── parse_hills.py         # HILLS parser and bias accumulation
│   └── build_dataset.py       # Merges CV trajectories with FES labels
│
├── models/
│   ├── train_gbr.py           # Gradient boosting regressor (scikit-learn)
│   ├── train_nn.py            # Feedforward neural network (PyTorch)
│   ├── hyperparameter_opt.py  # Optuna-based hyperparameter search
│   └── evaluate.py            # MAE, RMSE, R², parity plots
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_importance.ipynb
│   ├── 03_fes_comparison.ipynb   # Predicted vs. reference FES contour plots
│   └── 04_model_benchmarking.ipynb
│
├── scripts/
│   ├── run_metad.sh           # SLURM submission script for metadynamics
│   └── run_analysis.sh        # End-to-end pipeline from trajectory → features → model
│
├── results/
│   ├── figures/               # FES plots, parity plots, learning curves
│   └── model_checkpoints/     # Saved model weights (.pkl, .pt)
│
├── environment.yml            # Conda environment
├── requirements.txt
└── README.md
```

---

## Workflow

```
MD Trajectory (.xtc)          PLUMED WT-MetaD output
        │                              │
        ▼                              ▼
  Feature extraction            HILLS → FES (sum_hills)
  (distances, Rg, contacts)     ΔG on CV grid
        │                              │
        └──────────────┬───────────────┘
                       ▼
               Labelled dataset
               [features | ΔG]
                       │
                       ▼
              ML model training
          (GBR / Neural Network)
                       │
                       ▼
         Predict FES on new CV points
         without running metadynamics
```

---

## Installation

```bash
git clone https://github.com/your-username/ml-free-energy-predictor.git
cd ml-free-energy-predictor

# Create conda environment
conda env create -f environment.yml
conda activate ml-fes

# Or with pip
pip install -r requirements.txt
```

**Dependencies:** `MDAnalysis`, `numpy`, `pandas`, `scikit-learn`, `torch`, `optuna`, `matplotlib`, `seaborn`, `plumed` (Python wrapper)

---

## Quick Start

### 1. Extract features from trajectory

```bash
python features/extract_features.py \
  --topology  data/trajectories/system.tpr \
  --trajectory data/trajectories/traj.xtc \
  --output     data/features.csv
```

### 2. Parse metadynamics output and build dataset

```bash
python features/parse_hills.py \
  --hills data/hills/HILLS \
  --colvar data/colvar/COLVAR \
  --output data/fes_reference/fes.dat

python features/build_dataset.py \
  --features data/features.csv \
  --fes      data/fes_reference/fes.dat \
  --output   data/dataset.csv
```

### 3. Train the model

```bash
# Gradient boosting (fast baseline)
python models/train_gbr.py --data data/dataset.csv --output results/model_checkpoints/

# Neural network
python models/train_nn.py --data data/dataset.csv --epochs 200 --output results/model_checkpoints/
```

### 4. Evaluate and visualize

```bash
python models/evaluate.py \
  --model results/model_checkpoints/best_model.pt \
  --test  data/dataset.csv \
  --output results/figures/
```

---

## Results

| Model | MAE (kJ/mol) | RMSE (kJ/mol) | R² |
|---|---|---|---|
| Gradient Boosting | 1.82 | 2.41 | 0.94 |
| Neural Network (3-layer) | 1.34 | 1.87 | 0.97 |
| Neural Network + hyperopt | **0.98** | **1.45** | **0.98** |

The optimized neural network reproduces the reference free energy surface with sub-kT accuracy across the sampled CV space.

> Predicted vs. reference FES plots and learning curves are in `results/figures/`.

---

## Data Availability

Raw trajectories are not tracked due to file size. Processed feature matrices and reference FES grids used for model training are available on Zenodo:

> **DOI:** `10.5281/zenodo.XXXXXXX` *(will be updated upon publication)*

To reproduce the full pipeline from scratch, simulation input files (`.mdp`, `plumed.dat`, topology) are provided in `data/simulation_inputs/`.

---

## Citation

If you use this code or pipeline, please cite:

```bibtex
@article{yadav2026siRNA,
  author  = {Yadav, Desh Deepak and Bhandary, Debdip and Paik, Pradip},
  title   = {Molecular investigation to quantify the translocation of siRNA-NAPA
             oligomer complexes through lipid membrane},
  journal = {Physical Chemistry Chemical Physics},
  year    = {2026},
  doi     = {10.1039/D6CP00760K}
}
```

---

## Author

**Desh Deepak Yadav**  
PhD Researcher, Computational Biophysics  
School of Biomedical Engineering, IIT (BHU) Varanasi  
📧 ddesh5208660@gmail.com  
🔗 [ORCID](https://orcid.org/0000-0002-2513-8853) · [Google Scholar](#)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
