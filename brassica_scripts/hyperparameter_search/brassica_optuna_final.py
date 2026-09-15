"""
Unified Optuna hyperparameter search for XGBoost B. rapa subgenome dominance models.

USAGE
-----
Set MODEL_KEY via environment variable before submitting to HPC.
Valid values: "lf_mf1_all", "lf_mf1_groupI",
              "lf_mf2_all", "lf_mf2_groupI",
              "mf1_mf2_all", "mf1_mf2_groupI"

Example SLURM export:
    #SBATCH --export=MODEL_KEY=lf_mf1_all

METHODS SUMMARY
---------------------------------
Hyperparameters were tuned using Optuna (Akiba et al. 2019) with a TPE sampler.
For each trial, 10-fold stratified cross-validation was performed matching the
final model training procedure exactly: within each training fold, a stratified
10% inner validation split was carved out solely to govern XGBoost early stopping
(early_stopping_rounds=30); the test fold was never exposed to any training or
stopping decision. The objective metric was binary F1 (pos_label=1, i.e., the
non-dominant subgenome / WGD=1), averaged across all 10 folds. NaN values were
retained in the feature matrix for tau (undefined for genes with FPKM=0 across
all tissues), upstream_distance, and downstream_distance (undefined for genes at
chromosome boundaries); these were routed natively by XGBoost at each split.

Six models were tuned independently, one per pairwise subgenome comparison
(LF–MF1, LF–MF2, MF1–MF2) × dataset scope (all pairs, Group I arm–arm pairs
only). The 'location' feature (chromosomal arm vs. pericentromeric, encoded as
1.0/0.0) was excluded from Group I models because all Group I pairs are arm–arm,
making the feature constant and uninformative; this mirrors the treatment of
'location' in the analogous maize Group I model.

A two-tier search space was applied based on training-fold size:

  LARGE tier (lf_mf1_all: ~4,252; lf_mf1_groupI: ~3,665;
              lf_mf2_all: ~3,587 training genes/fold):
    max_depth ceiling = 10, n_trials = 400

  SMALL tier (lf_mf2_groupI: ~2,858; mf1_mf2_all: ~2,324;
              mf1_mf2_groupI: ~1,831 training genes/fold):
    max_depth ceiling = 8,  n_trials = 600

The max_depth ceiling for the LARGE tier was set to 10 rather than the
maize-equivalent 12 because the largest Brassica training folds (~4,252
genes/fold) are smaller than the analogous maize LARGE folds (~6,350
and ~3,860 genes/fold), reducing the training data available to support
very deep trees without overfitting. The SMALL tier ceiling of 8 is
identical to maize. min_child_weight was searched over 1–15 for both
tiers (vs. 1–10 in maize) to provide additional regularization headroom
given the consistently smaller Brassica training sets. All other search
space bounds are identical to those used for the maize models.

The MF1–MF2 models compare two already-fractionated subgenomes, where
the dominance signal is expected to be weaker than in LF-involving
comparisons. The SMALL tier's increased trial budget (600 vs. 400)
partially compensates for the noisier per-fold F1 estimates that arise
from smaller test folds in these comparisons.
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# ============================================================
# GLOBAL CONFIGURATION
# Set MODEL_KEY via environment variable before submitting.
# Valid values: "lf_mf1_all", "lf_mf1_groupI",
#               "lf_mf2_all", "lf_mf2_groupI",
#               "mf1_mf2_all", "mf1_mf2_groupI"
# ============================================================

MODEL_KEY = os.environ.get("MODEL_KEY", None)
assert MODEL_KEY is not None, (
    "MODEL_KEY environment variable not set. Set via SLURM export, e.g.:\n"
    "  #SBATCH --export=MODEL_KEY=lf_mf1_all"
)

# ============================================================
# PATHS
# ============================================================
FEATURESETS_DIR = Path("../final_featuresets")

INPUT_PATHS = {
    "lf_mf1_all":     FEATURESETS_DIR / "Brapa_lf_mf1_final.csv",
    "lf_mf1_groupI":  FEATURESETS_DIR / "Brapa_lf_mf1_groupI_final.csv",
    "lf_mf2_all":     FEATURESETS_DIR / "Brapa_lf_mf2_final.csv",
    "lf_mf2_groupI":  FEATURESETS_DIR / "Brapa_lf_mf2_groupI_final.csv",
    "mf1_mf2_all":    FEATURESETS_DIR / "Brapa_mf1_mf2_final.csv",
    "mf1_mf2_groupI": FEATURESETS_DIR / "Brapa_mf1_mf2_groupI_final.csv",
}

# Kept local and regenerable — not tracked in git (see .gitignore).
OUTPUT_DIR = Path("../brassica_outputs/optuna_results")

# ============================================================
# MODEL-LEVEL SETTINGS (automatically derived from MODEL_KEY)
# ============================================================
#
# Two-tier search space based on training-fold size (0.9 × n_genes):
#
#   LARGE tier — lf_mf1_all (~4,252), lf_mf1_groupI (~3,665),
#                lf_mf2_all (~3,587):
#     max_depth ceiling = 10, n_trials = 400
#
#   SMALL tier — lf_mf2_groupI (~2,858), mf1_mf2_all (~2,324),
#                mf1_mf2_groupI (~1,831):
#     max_depth ceiling = 8,  n_trials = 600
#
# The tier boundary sits between lf_mf2_all and lf_mf2_groupI
# (~730-gene gap), which is also the natural split between LF-dominant
# comparisons and the smaller MF1–MF2 comparisons.
#
# min_child_weight searched over 1–15 for both tiers. Upper bound is
# wider than maize (1–10) to provide regularization headroom for the
# consistently smaller Brassica training sets.
#
# All other search space bounds are identical to the maize Optuna script.

LARGE_MODELS = {"lf_mf1_all", "lf_mf1_groupI", "lf_mf2_all"}
SMALL_MODELS  = {"lf_mf2_groupI", "mf1_mf2_all", "mf1_mf2_groupI"}

assert MODEL_KEY in INPUT_PATHS, (
    f"Invalid MODEL_KEY '{MODEL_KEY}'.\n"
    f"Valid values: {sorted(INPUT_PATHS.keys())}"
)

N_TRIALS          = 400 if MODEL_KEY in LARGE_MODELS else 600
MAX_DEPTH_CEILING = 10  if MODEL_KEY in LARGE_MODELS else 8

# Derive PAIR and USE_GROUP1 from MODEL_KEY.
# USE_GROUP1=True  => arm-arm pairs only; 'location' is constant (all 1.0)
#                     and is excluded from the feature matrix.
# USE_GROUP1=False => all pairs; 'location' is informative and included.
USE_GROUP1   = MODEL_KEY.endswith("groupI")
USE_LOCATION = not USE_GROUP1

# Derive PAIR label (used in print output and output filenames)
PAIR = MODEL_KEY.replace("_all", "").replace("_groupI", "")

os.makedirs(OUTPUT_DIR, exist_ok=True)
input_path = INPUT_PATHS[MODEL_KEY]

print("=" * 60)
print(f"MODEL_KEY          : {MODEL_KEY}")
print(f"PAIR               : {PAIR}")
print(f"USE_GROUP1         : {USE_GROUP1}")
print(f"USE_LOCATION       : {USE_LOCATION}")
print(f"N_TRIALS           : {N_TRIALS}")
print(f"MAX_DEPTH_CEILING  : {MAX_DEPTH_CEILING}")
print(f"Input path         : {input_path}")
print(f"Output dir         : {OUTPUT_DIR}")
print("=" * 60)
print()

# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    """
    Loads the stacked (gene-level) feature matrix for the current MODEL_KEY.

    The input CSVs are produced by the wide-to-long stack_pairwise() function
    in brapaMLpreprocess_Final_genepairs.ipynb. Each row represents one gene;
    subgenome suffixes (_LF, _MF1, _MF2) are stripped during stacking so all
    column names here are unsuffixed (e.g., 'tau', not 'tau_LF').

    NaN values are intentionally retained in X for three columns:
      - tau              : undefined (not missing) when FPKM=0 across all
                           tissues; retaining NaN preserves this biological
                           distinction from low-expression genes.
      - upstream_distance : undefined for genes within 500 bp of a chromosome
                            boundary where the ACR upstream window cannot be
                            fully extracted — a positional artifact.
      - downstream_distance: same as upstream_distance.
    XGBoost handles NaN natively via learned missing-value routing at each
    split. Dropping these rows would remove biologically interpretable edge
    cases from hyperparameter tuning.

    Exclusion logic:
      - 'WGD'      : always excluded — it is the label vector (y).
      - 'location' : excluded when USE_LOCATION=False (Group I models),
                     because all Group I pairs are arm-arm, making the
                     feature constant (1.0) and uninformative.
      - 'rec_rate' : always included in Optuna tuning. The ablation study
                     (USE_RECOMB=False) is run in the modeling notebook using
                     the tuned hyperparameters — identical to the maize approach.

    Feature columns are defined by name, not position, to guard against
    column-order differences across input files.

    Returns
    -------
    X            : np.ndarray — feature matrix, shape (n_genes, n_features)
    y            : np.ndarray — label vector (WGD), shape (n_genes,)
    feature_names: list[str]  — ordered feature names corresponding to X columns
    """
    my_data = pd.read_csv(input_path)

    # --- Required column checks ---
    assert "WGD" in my_data.columns, \
        "CRITICAL: 'WGD' column not found. Check input file."
    assert "location" in my_data.columns, \
        "CRITICAL: 'location' column not found. Check input file."
    assert "rec_rate" in my_data.columns, \
        "CRITICAL: 'rec_rate' column not found. Check input file."
    assert "tau" in my_data.columns, \
        "CRITICAL: 'tau' column not found. Check input file."
    assert "upstream_distance" in my_data.columns, \
        "CRITICAL: 'upstream_distance' column not found. Check input file."
    assert "downstream_distance" in my_data.columns, \
        "CRITICAL: 'downstream_distance' column not found. Check input file."

    # --- Build exclusion set ---
    exclude = {"WGD"}
    if not USE_LOCATION:
        exclude.add("location")
    # rec_rate is always included for Optuna tuning

    feature_names = [col for col in my_data.columns if col not in exclude]

    X = my_data[feature_names].to_numpy()
    y = my_data["WGD"].to_numpy()

    # --- Sanity checks ---
    assert X.shape[0] == y.shape[0], \
        f"CRITICAL: Row mismatch — X: {X.shape[0]} rows, y: {y.shape[0]} rows."
    assert set(np.unique(y)) == {0, 1}, \
        f"CRITICAL: Unexpected label values: {np.unique(y)}. Expected {{0, 1}}."

    # WGD balance check: stacking always produces exactly equal class counts
    n_wgd0 = (y == 0).sum()
    n_wgd1 = (y == 1).sum()
    assert n_wgd0 == n_wgd1, (
        f"CRITICAL: WGD class imbalance detected — WGD=0: {n_wgd0}, "
        f"WGD=1: {n_wgd1}. The stacked dataset should always be balanced."
    )

    # Location check for Group I: must be constant (all 1.0) if excluded
    if not USE_LOCATION:
        loc_vals = my_data["location"].dropna().unique()
        if not (len(loc_vals) == 1 and loc_vals[0] == 1.0):
            raise ValueError(
                f"CRITICAL: Group I model '{MODEL_KEY}' has non-constant "
                f"location values {loc_vals}. Expected all 1.0 (arm-arm only). "
                f"Check group assignment in preprocessing."
            )

    print(f"Loaded data shape  : {my_data.shape}")
    print(f"Features used      : {len(feature_names)}")
    print(f"Label distribution : WGD=0 (dominant): {n_wgd0}, "
          f"WGD=1 (non-dominant): {n_wgd1}")

    # --- NaN report for exempt columns ---
    nan_exempt = {"tau", "upstream_distance", "downstream_distance"}
    nan_counts = pd.DataFrame(X, columns=feature_names).isna().sum()
    nan_counts = nan_counts[nan_counts > 0]

    if len(nan_counts) > 0:
        print("\nNaN values retained (XGBoost routes natively):")
        for col, n in nan_counts.items():
            if col in nan_exempt:
                print(f"  {col}: {n} NaNs  [biologically exempt]")
            else:
                print(f"  {col}: {n} NaNs  [WARNING: unexpected — check dropout]")
    else:
        print("No NaN values in feature matrix.")
    print()

    return X, y, feature_names


X, y, feature_names = load_data()

# ============================================================
# OPTUNA OBJECTIVE
# ============================================================
def objective(trial):
    """
    Evaluates a hyperparameter configuration using 10-fold stratified CV.

    Training setup mirrors the modeling notebook (test_classifier()) exactly:
      - StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
      - Inner val split: train_test_split(test_size=0.1, stratify=y_train_full,
        random_state=42) carved from each training fold — used only for early
        stopping. The test fold is never exposed to any fitting or stopping
        decision.
      - eval_set = [(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=30

    Objective metric: binary F1 (pos_label=1 / WGD=1 / non-dominant
    subgenome), averaged across all 10 folds. Matches f1_score() as
    reported in the modeling notebook.

    Differences from maize Optuna script:
      - MAX_DEPTH_CEILING: 10 (LARGE) or 8 (SMALL) vs. 12/8 in maize
      - min_child_weight upper bound: 15 vs. 10 in maize
      - N_TRIALS: 400 (LARGE) or 600 (SMALL) vs. 300/500 in maize
      - All other bounds are identical.
    """
    params = {
        # log=True samples learning rate on a log scale, giving finer
        # resolution at small values where behaviour differs most.
        "learning_rate":         trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        # n_estimators acts as a ceiling; early stopping halts training
        # earlier if validation logloss plateaus.
        "n_estimators":          trial.suggest_int("n_estimators", 100, 1000),
        # max_depth ceiling is tier-dependent (10 LARGE, 8 SMALL).
        "max_depth":             trial.suggest_int("max_depth", 3, MAX_DEPTH_CEILING),
        "subsample":             trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":      trial.suggest_float("colsample_bytree", 0.5, 1.0),
        # Upper bound of 15 (vs. 10 in maize) to provide regularization
        # headroom for smaller Brassica training sets.
        "min_child_weight":      trial.suggest_int("min_child_weight", 1, 15),
        # gamma: minimum loss reduction to make a split; controls tree complexity.
        "gamma":                 trial.suggest_float("gamma", 0.0, 5.0),
        # L1 regularization on leaf weights.
        "reg_alpha":             trial.suggest_float("reg_alpha", 0.0, 5.0),
        # L2 regularization on leaf weights; lower bound 0.1 (XGBoost default 1.0).
        "reg_lambda":            trial.suggest_float("reg_lambda", 0.1, 15.0),
        "eval_metric":           "logloss",
        "early_stopping_rounds": 30,
        "random_state":          42,
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    f1_scores = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):

        # Outer split: test fold held out entirely — never used for
        # fitting or stopping decisions.
        X_train_full, X_test = X[train_index], X[test_index]
        y_train_full, y_test = y[train_index], y[test_index]

        # Inner split: carve 10% of training fold as validation set for
        # early stopping only. Mirrors modeling notebook: train_test_split(
        # test_size=0.1, stratify=y_train_full, random_state=42).
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.1,
            random_state=42,
            stratify=y_train_full
        )

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        # Predict on held-out test fold.
        # predict() uses best_iteration internally, not the final round.
        y_pred = model.predict(X_test)

        # Binary F1: pos_label=1 (WGD=1 / non-dominant subgenome).
        # Matches modeling notebook's reported metric.
        fold_f1 = f1_score(y_test, y_pred, average="binary", pos_label=1)
        f1_scores.append(fold_f1)

    return np.mean(f1_scores)


# ============================================================
# OPTUNA CALLBACK — prints trial-level progress to stdout.
# Useful for monitoring SLURM logs without per-fold verbosity.
# ============================================================
def print_callback(study, trial):
    if trial.number % 10 == 0 or trial.number == N_TRIALS - 1:
        print(f"  Trial {trial.number:>4} | "
              f"F1: {trial.value:.4f} | "
              f"Best so far: {study.best_value:.4f} "
              f"(trial {study.best_trial.number})")


# ============================================================
# RUN OPTIMIZATION
# ============================================================
print(f"Starting Optuna: {N_TRIALS} trials | Model: {MODEL_KEY} | "
      f"max_depth ceiling: {MAX_DEPTH_CEILING} | "
      f"min_child_weight range: 1–15")
print("-" * 60)

# Suppress Optuna's internal logging — progress handled by print_callback.
optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, callbacks=[print_callback])

# ============================================================
# RESULTS
# ============================================================
best_params = study.best_params
best_f1     = study.best_value

print()
print("=" * 60)
print(f"OPTUNA COMPLETE — {MODEL_KEY}")
print("=" * 60)
print(f"Best binary F1 (mean, 10-fold CV) : {best_f1:.4f}")
print(f"Best trial number                 : {study.best_trial.number}")
print(f"Completed trials                  : {len(study.trials)}")
print()
print("Best hyperparameters:")
for k, v in best_params.items():
    if isinstance(v, float):
        print(f"  {k:<22}: {v:.8f}")
    else:
        print(f"  {k:<22}: {v}")

# Print a block formatted for direct copy-paste into the modeling notebook's
# hyperparams dict. Pair and USE_GROUP1 are encoded as the tuple key.
_use_group1_str = "True" if USE_GROUP1 else "False"
print()
print("--- Copy-paste block for notebook hyperparams dict ---")
print(f'    ("{PAIR}", {_use_group1_str}): dict(')
print(f'        eval_metric="logloss",')
for k, v in best_params.items():
    if isinstance(v, float):
        print(f'        {k}={v:.8f},')
    else:
        print(f'        {k}={v},')
print(f'        early_stopping_rounds=30,')
print(f'    ),')
print("------------------------------------------------------")

# ============================================================
# SAVE TO JSON
# ============================================================
output_record = {
    "model_key":           MODEL_KEY,
    "pair":                PAIR,
    "use_group1":          USE_GROUP1,
    "use_location":        USE_LOCATION,
    "best_binary_f1":      best_f1,
    "best_trial_number":   study.best_trial.number,
    "n_trials_completed":  len(study.trials),
    "n_folds":             10,
    "tier":                "LARGE" if MODEL_KEY in LARGE_MODELS else "SMALL",
    "max_depth_ceiling":   MAX_DEPTH_CEILING,
    "n_features":          len(feature_names),
    "best_params":         best_params,
    # Fixed params not searched by Optuna — recorded for completeness
    "fixed_params": {
        "eval_metric":           "logloss",
        "early_stopping_rounds": 30,
        "random_state":          42,
    },
    # Search space bounds — recorded so the methods section can cite them
    # precisely without referring back to the script.
    "search_space": {
        "learning_rate":    [0.001, 0.3, "log"],
        "n_estimators":     [100, 1000],
        "max_depth":        [3, MAX_DEPTH_CEILING],
        "subsample":        [0.5, 1.0],
        "colsample_bytree": [0.5, 1.0],
        "min_child_weight": [1, 15],
        "gamma":            [0.0, 5.0],
        "reg_alpha":        [0.0, 5.0],
        "reg_lambda":       [0.1, 15.0],
    },
}

out_path = os.path.join(OUTPUT_DIR, f"optuna_{MODEL_KEY}_results.json")
with open(out_path, "w") as f:
    json.dump(output_record, f, indent=4)

print(f"\nFull results saved to: {out_path}")