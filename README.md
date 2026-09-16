# Subgenome Dominance Predictor

This repository contains the data preprocessing, machine learning, and explainable
AI (XAI) analysis pipeline for *"Explainable AI identifies recombination and
chromatin environment as key predictors of subgenome evolution in maize and
Brassica"*. The project integrates genomic and epigenomic features — including
recombination rate, gene expression, evolutionary distance, GC content,
transposable elements (TEs), DNA methylation, histone modifications, and
accessible chromatin regions (ACRs) — to classify genes by subgenome of origin
in two independent polyploid systems, maize (*Zea mays*) and *Brassica rapa*,
and to interpret the resulting models using SHAP-based XAI methods.

## Repository Structure
subgen_dominator/
├── maize_scripts/ Maize analysis pipeline
│ ├── feature_preprocessing/ Per-feature preprocessing scripts (GC content, expression, ACR)
│ ├── hyperparameter_search/ Optuna hyperparameter tuning
│ └── *.ipynb, *.py Main pipeline: preprocessing, correlations, MWU,
│ benchmarking, modeling/SHAP, networks, ablation,
│ summary figures, blob analysis
├── brassica_scripts/ Brassica analysis pipeline (mirrors maize_scripts/)
├── final_featuresets/ Model-ready feature CSVs (committed directly — see below)
├── inputData/ Placeholder for raw/intermediate input data (see Data Availability)
├── maize_outputs/ Local, regenerable analysis outputs (gitignored — see below)
├── brassica_outputs/ Local, regenerable analysis outputs (gitignored — see below)
├── requirements.txt
└── README.md

## Data Availability

Raw and intermediate feature-source files for both species are archived on Zenodo.

**Zenodo DOI:** 10.5281/zenodo.22797527

### Setup
1. Download the archive from the DOI above.
2. Extract its contents into the `inputData/` folder at the repository root,
   preserving the folder names below.
   
inputData/
├── maize_inputData/ Raw/intermediate feature files for maize
└── brapa_inputData/ Raw/intermediate feature files for B. rapa

See `inputData/README.md` for the full expected structure.

### What's already in this repository (no download needed)
Model-ready feature sets are included directly under `final_featuresets/`.

### What's regenerated locally, not archived anywhere
Several categories of output are neither committed to GitHub nor archived
on Zenodo, because they can be fully and exactly reproduced by re-running
the code in this repository against the data above:

- **Model checkpoints** (`*_cv_checkpoint.npz`, `*_shap_by_label.npz`) —
  outputs of the XGBoost modeling notebooks. These are large (up to ~85 MB
  per file) and fully reproducible from fixed random seeds, tuned
  hyperparameters (see `hyperparameter_search/`), and the committed feature
  sets. Set `RETRAIN_MODEL = True` in the relevant notebook to regenerate.
- **Analysis outputs** (correlation tables/heatmaps, MWU results, network
  figures, ablation statistics, benchmarking tables, blob analysis, Optuna
  search results) — written to `maize_outputs/`/`brassica_outputs/` when
  the corresponding notebook or script is run; not tracked in git (see
  `.gitignore`).
  
## Setup

### Requirements
- Python 3.9+ (developed and tested on Python 3.11.4)
- Conda (recommended) or another virtual environment tool

### Installation

```bash
# Create and activate a new environment
conda create -n subgen_dominator python=3.11.4
conda activate subgen_dominator

# Install dependencies
pip install -r requirements.txt
```

All exploratory and pipeline code is provided as Jupyter notebooks (`.ipynb`)
and standalone Python scripts (`.py`). Notebooks can be run via
`jupyter notebook` or `jupyter lab` from the repository root, or opened
directly in an IDE that supports the Jupyter protocol (e.g., VS Code,
JupyterLab, PyCharm).

Some scripts in `feature_preprocessing/` and `hyperparameter_search/` are
designed for command-line or HPC/SLURM execution rather than interactive
use — see the docstring at the top of each script for usage instructions.

## Reproducing the Analysis

Each species' pipeline should be run in the order below. Steps marked
**"run once per GROUP"** require manually setting a configuration variable
at the top of the notebook and re-running it for each value. This is
noted explicitly in each notebook's own configuration cell.

### Maize

1. **Preprocessing** — `maize_scripts/maizeMLpreprocess.ipynb`
   Merges all genomic/epigenomic feature categories and produces the
   model-ready feature sets in `final_featuresets/` (already committed;
   re-running this step is only necessary to reproduce from raw data).

2. **Feature correlations** — `maize_scripts/maize_correlations.py <GROUP>`
   Run once per group: `All`, `I`, `II`, `III`, `IV`.
   Produces Spearman correlation heatmaps and tables (Data S2) in
   `maize_outputs/corr_output/`.

3. **Mann-Whitney U analysis** — `maize_scripts/maize_mwu_analysis.ipynb`
   Processes all five groups in a single run. Produces MWU test results
   and volcano/lollipop plots (Data S1) in `maize_outputs/mwu_output/`.

4. **Model benchmarking**
   - `maize_scripts/maizeMLModels_eval_set.ipynb` — run once per group,
     evaluating four baseline classifiers (LR, SVM, RF, XGBoost) with
     default hyperparameters.
   - `maize_scripts/maize_benchmarking_table.ipynb` — run once, after all
     five groups above are complete. Aggregates results and performs
     Wilcoxon signed-rank significance testing (Table S1).

5. **Final XGBoost modeling and SHAP analysis** —
   `maize_scripts/maizeMLModels.ipynb`
   Run once per group, with `USE_RECOMB = True`. Set `RETRAIN_MODEL = True`
   the first time (or after any config change); set to `False` afterward to
   reload from checkpoint. Uses Optuna-tuned hyperparameters (see Step 9).
   Produces SHAP values, beeswarm plots (Fig. 3), and model checkpoints in
   `maize_outputs/model_outputs/`.

6. **Summary figures** — `maize_scripts/maizeMLModels_summary_figs.ipynb`
   Run once, after Step 5 is complete for all five groups. Produces the
   combined F1/ROC panels (Fig. 1E, 1G).

7. **SHAP co-variation and interaction networks** —
   `maize_scripts/networks_maize.ipynb`
   Run once; reads SHAP data for all five groups. Produces network figures
   (Figs. 4–5) and edge/degree tables (Data S4–S6) in
   `maize_outputs/network_output/`.

8. **Recombination SHAP blob analysis** —
   `maize_scripts/maize_recomb_shap_blob_analysis.ipynb`
   Run once. Requires Step 5's Group I output. Produces the Fisher's exact
   and Mann-Whitney U blob comparisons (Data S3) discussed in the Results
   text, in `maize_outputs/blob_analysis/`.

9. **Recombination rate ablation study**
   - `maize_scripts/maizeMLModels.ipynb` — re-run once per group with
     `USE_RECOMB = False` (no-recombination condition).
   - `maize_scripts/maizeMLModels_recomb_only.ipynb` — run once per group
     (recombination-only condition).
   - `maize_scripts/maize_ablation_significance_tests.ipynb` — run once,
     after both conditions above are complete for all groups. Produces
     Fig. 8, Fig. S92, and Data S8.

10. **Hyperparameter tuning (optional)** —
    `maize_scripts/hyperparameter_search/maize_optuna_final.py`
    Not required to reproduce reported results — tuned values are already
    hardcoded in the notebooks above (Tables S2–S3). Provided for
    transparency and designed for SLURM submission; run once per group via
    the `GROUP` environment variable.
    
### *Brassica rapa*

The Brassica pipeline mirrors the maize pipeline, with two structural
differences: instead of a single dataset processed across 5 chromosomal
groups, Brassica has 3 pairwise subgenome comparisons (LF–MF1, LF–MF2,
MF1–MF2) × 2 dataset scopes (all pairs, or Group I arm–arm pairs only) — 6
combinations total, controlled by `PAIR` and `USE_GROUP1` (or `MODEL_KEY`
where noted).

1. **Preprocessing** — `brassica_scripts/brapaMLpreprocess_Final_genepairs.ipynb`
   Produces the model-ready feature sets in `final_featuresets/` (already
   committed).

2. **Feature correlations** — `brassica_scripts/brassica_correlations.py <MODEL_KEY>`
   Run once per `MODEL_KEY`: `lf_mf1_all`, `lf_mf1_groupI`, `lf_mf2_all`,
   `lf_mf2_groupI`, `mf1_mf2_all`, `mf1_mf2_groupI`.
   Produces Data S2 in `brassica_outputs/corr_output/`.

3. **Mann-Whitney U analysis** — `brassica_scripts/brassica_mwu_analysis.ipynb`
   Processes all six combinations in a single run. Produces Data S1 in
   `brassica_outputs/mwu_output/`.

4. **Model benchmarking**
   - `brassica_scripts/brassicaMLModels_eval_set.ipynb` — run once per
     `PAIR`/`USE_GROUP1` combination (6 total).
   - `brassica_scripts/brassica_benchmarking_table.ipynb` — run once, after
     all six above are complete (Table S4).

5. **Final XGBoost modeling and SHAP analysis** —
   `brassica_scripts/brassicaMLModels.ipynb`
   Run once per `PAIR`/`USE_GROUP1` combination, with `USE_RECOMB = True`
   (6 runs total). Same `RETRAIN_MODEL` checkpoint behavior as maize.
   Produces Fig. 6 and model checkpoints in `brassica_outputs/model_outputs/`.

6. **Summary figures** — `brassica_scripts/brassicaMLModels_summary_figs.ipynb`
   Run once, after Step 5 is complete for all six combinations. Produces
   Fig. 1F, 1H.

7. **SHAP co-variation and interaction networks** —
   `brassica_scripts/networks_brassica.ipynb`
   Run once; reads SHAP data for all six combinations. Produces Fig. 7 and
   Data S4–S7 in `brassica_outputs/network_output/`.

8. **Recombination rate ablation study**
   - `brassica_scripts/brassicaMLModels.ipynb` — re-run once per
     `PAIR`/`USE_GROUP1` combination with `USE_RECOMB = False`.
   - `brassica_scripts/brassicaMLModels_recomb_only.ipynb` — run once per
     combination.
   - `brassica_scripts/brassica_ablation_significance_tests.ipynb` — run
     once, after both conditions above are complete for all six
     combinations. Produces Fig. 8, Fig. S93, and Data S8.

9. **Hyperparameter tuning (optional)** —
   `brassica_scripts/hyperparameter_search/brassica_optuna_final.py`
   Not required to reproduce reported results (Table S5). Designed for
   SLURM submission; run once per `MODEL_KEY` via environment variable.
   
## Citation

If you use this code or data, please cite:

> Schuster, L.A., Liu, B., Silva, J.C.F., Cheng, X., Dias, R., Zhao, M.
> Explainable AI identifies recombination and chromatin environment as key
> predictors of subgenome evolution in maize and *Brassica*. *Science
> Advances* (accepted). DOI: TBD

If you use the archived data, please also cite the Zenodo deposit:

> [Zenodo citation — TBD once the DOI is minted]

## Contact

For questions about this repository, contact the corresponding authors:
raquel.dias@ufl.edu, meixiazhao@ufl.edu