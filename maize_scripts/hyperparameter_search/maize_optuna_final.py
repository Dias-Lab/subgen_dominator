"""
Unified Optuna hyperparameter search for XGBoost subgenome dominance models.

USAGE
-----
Set GROUP via environment variable before submitting to HPC.
Valid values: "All", "I", "II", "III", "IV"

Example SLURM export:
    #SBATCH --export=GROUP=All

METHODS SUMMARY
---------------------------------
Hyperparameters were tuned using Optuna (Akiba et al. 2019) with a TPE sampler.
For each trial, 10-fold stratified cross-validation was performed matching the
final model training procedure exactly: within each training fold, a stratified
10% inner validation split was carved out solely to govern XGBoost early stopping
(early_stopping_rounds=30); the test fold was never exposed to any training or
stopping decision. The objective metric was binary F1 (pos_label=1, i.e., Maize2 /
WGD=1), averaged across all 10 folds. NaN values were retained in the feature
matrix and routed natively by XGBoost at each split. The 'location' feature was
included only for the All model (60 features), consistent with the final models.
Search space for max_depth was capped at 8 for smaller groups (II, III, IV) to
reduce overfitting risk given smaller training set sizes (~500-900 samples per fold).
Groups II, III, and IV used 500 trials; All and Group I used 300 trials.
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
# Set GROUP before submitting to HPC.
# Valid values: "All", "I", "II", "III", "IV"
# ============================================================

GROUP = os.environ.get("GROUP", None)
assert GROUP is not None, "GROUP environment variable not set. Set via SLURM export."

# ============================================================
# PATHS
# ============================================================
FEATURESETS_DIR = Path("../final_featuresets")

INPUT_PATHS = {
    "All": FEATURESETS_DIR / "processedMaize_final.csv",
    "I":   FEATURESETS_DIR / "processedMaize_groupI_final.csv",
    "II":  FEATURESETS_DIR / "processedMaize_groupII_final.csv",
    "III": FEATURESETS_DIR / "processedMaize_groupIII_final.csv",
    "IV":  FEATURESETS_DIR / "processedMaize_groupIV_final.csv",
}

# Kept local and regenerable — not tracked in git (see .gitignore).
OUTPUT_DIR = Path("../maize_outputs/optuna_results")

# ============================================================
# GROUP-LEVEL SETTINGS (automatically derived from GROUP)
# ============================================================
# Two-tier search space based on training set size:
#
#   All (~6350 training samples/fold), Group I (~3860):
#     max_depth ceiling = 12, n_trials = 300
#
#   Group II (~588 training samples/fold), III (~940), IV (~730):
#     max_depth ceiling = 8 to reduce overfitting risk on small training sets.
#     n_trials = 500 to compensate for noisier per-fold F1 from small test folds.
#
# All other search space bounds are identical across groups.

LARGE_GROUPS = {"All", "I"}
SMALL_GROUPS  = {"II", "III", "IV"}

assert GROUP in INPUT_PATHS, (
    f"Invalid GROUP '{GROUP}'. Valid values: {list(INPUT_PATHS.keys())}"
)

N_TRIALS          = 300 if GROUP in LARGE_GROUPS else 500
MAX_DEPTH_CEILING = 12  if GROUP in LARGE_GROUPS else 8

# 'location' is included only for the All model.
# For groups I-IV it is either redundant with the label (III, IV)
# or dropped for cross-group consistency (I, II).
USE_LOCATION = (GROUP == "All")

os.makedirs(OUTPUT_DIR, exist_ok=True)
input_path = INPUT_PATHS[GROUP]

print("=" * 60)
print(f"GROUP              : {GROUP}")
print(f"Input path         : {input_path}")
print(f"USE_LOCATION       : {USE_LOCATION}")
print(f"N_TRIALS           : {N_TRIALS}")
print(f"MAX_DEPTH_CEILING  : {MAX_DEPTH_CEILING}")
print(f"Output dir         : {OUTPUT_DIR}")
print("=" * 60)
print()

# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    """
    Loads the feature matrix for the current GROUP.

    NaN values are intentionally retained in X:
      - tau:            NaN indicates FPKM = 0 across all tissues (biologically meaningful)
      - acr_Lup/Ldown:  NaN indicates a chromosome-edge gene with no upstream/downstream
                        flanking sequence available
    XGBoost handles NaN natively via learned missing-value routing at each split.
    Dropping these rows would remove biologically interpretable edge cases from tuning.

    Feature columns are defined by name, not position, to guard against
    column-order differences across input files.
    """
    my_data = pd.read_csv(input_path)

    assert "WGD" in my_data.columns, \
        "CRITICAL: 'WGD' column not found. Check input file."
    assert "location" in my_data.columns, \
        "CRITICAL: 'location' column not found. Check input file."

    if USE_LOCATION:
        feature_names = [col for col in my_data.columns if col != "WGD"]
    else:
        feature_names = [col for col in my_data.columns
                         if col not in ("WGD", "location")]

    X = my_data[feature_names].to_numpy()
    y = my_data["WGD"].to_numpy()

    assert X.shape[0] == y.shape[0], \
        f"CRITICAL: Row mismatch — X: {X.shape[0]} rows, y: {y.shape[0]} rows."
    assert set(np.unique(y)) == {0, 1}, \
        f"CRITICAL: Unexpected label values: {np.unique(y)}. Expected {{0, 1}}."

    print(f"Loaded data shape  : {my_data.shape}")
    print(f"Features used      : {len(feature_names)}")
    print(f"Label distribution : WGD=0 (Maize1): {(y == 0).sum()}, "
          f"WGD=1 (Maize2): {(y == 1).sum()}")

    nan_counts = pd.DataFrame(X, columns=feature_names).isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts) > 0:
        print("NaN values retained (XGBoost routes natively):")
        for col, n in nan_counts.items():
            print(f"  {col}: {n} NaNs")
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

    Training setup mirrors the notebook (test_classifier()) exactly:
      - StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
      - Inner val split: train_test_split(test_size=0.1, stratify=y_train_full,
        random_state=42) carved from each training fold — used only for early stopping
      - eval_set = [(X_train, y_train), (X_val, y_val)], early_stopping_rounds=30

    Objective metric: binary F1, pos_label=1 (WGD=1 / Maize2), averaged over 10 folds.
    This matches f1_score(y_true, y_pred_binary) as reported in the notebook.
    Binary F1 differs from micro-averaged F1: micro F1 equals accuracy in binary
    classification, whereas binary F1 is the harmonic mean of precision and recall
    for the positive class (WGD=1) only. Binary F1 was chosen to match the notebook's
    reported metric.
    """
    params = {
        # log=True samples learning rate on a log scale, giving more resolution
        # at small values where behavior differs most between rates
        "learning_rate":         trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        # Upper bound of 1000 with early stopping means n_estimators acts as a ceiling;
        # the model stops earlier if validation loss plateaus
        "n_estimators":          trial.suggest_int("n_estimators", 100, 1000),
        # max_depth ceiling is group-dependent (12 for large groups, 8 for small)
        "max_depth":             trial.suggest_int("max_depth", 3, MAX_DEPTH_CEILING),
        "subsample":             trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":      trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":      trial.suggest_int("min_child_weight", 1, 10),
        # gamma: minimum loss reduction to make a split; controls tree complexity
        "gamma":                 trial.suggest_float("gamma", 0.0, 5.0),
        # L1 regularization on weights
        "reg_alpha":             trial.suggest_float("reg_alpha", 0.0, 5.0),
        # L2 regularization on weights; lower bound 0.1 (XGBoost default is 1.0)
        "reg_lambda":            trial.suggest_float("reg_lambda", 0.1, 15.0),
        "eval_metric":           "logloss",
        "early_stopping_rounds": 30,
        "random_state":          42,
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    f1_scores = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):

        # Outer split: test fold held out entirely — never used for stopping or fitting
        X_train_full, X_test = X[train_index], X[test_index]
        y_train_full, y_test = y[train_index], y[test_index]

        # Inner split: carve 10% of training fold as validation set for early stopping.
        # Mirrors notebook: train_test_split(test_size=0.1, stratify=y_train_full,
        # random_state=42). This also matches the background data used for
        # TreeExplainer in the final models (X_train, not X_train_full).
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
        # predict() internally uses best_iteration — not the final boosting round.
        y_pred = model.predict(X_test)

        # Binary F1: pos_label=1 (WGD=1 / Maize2).
        # Matches notebook's reported metric: f1_score(y_true, y_pred_binary).
        fold_f1 = f1_score(y_test, y_pred, average="binary", pos_label=1)
        f1_scores.append(fold_f1)

    return np.mean(f1_scores)


# ============================================================
# OPTUNA CALLBACK — prints trial-level progress to stdout
# Useful for monitoring SLURM logs without per-fold verbosity.
# ============================================================
def print_callback(study, trial):
    # Print every 10 trials and on the last trial
    if trial.number % 10 == 0 or trial.number == N_TRIALS - 1:
        print(f"  Trial {trial.number:>4} | "
              f"F1: {trial.value:.4f} | "
              f"Best so far: {study.best_value:.4f} "
              f"(trial {study.best_trial.number})")


# ============================================================
# RUN OPTIMIZATION
# ============================================================
print(f"Starting Optuna: {N_TRIALS} trials | Group {GROUP} | "
      f"max_depth ceiling: {MAX_DEPTH_CEILING}")
print("-" * 60)

# Suppress Optuna's internal logging — progress handled by print_callback above
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
print(f"OPTUNA COMPLETE — Group {GROUP}")
print("=" * 60)
print(f"Best binary F1 (mean, 10-fold CV) : {best_f1:.4f}")
print(f"Best trial number                 : {study.best_trial.number}")
print(f"Completed trials                  : {len(study.trials)}")
print()
print("Best hyperparameters:")
for k, v in best_params.items():
    # Format floats consistently; ints printed as-is
    if isinstance(v, float):
        print(f"  {k:<22}: {v:.8f}")
    else:
        print(f"  {k:<22}: {v}")

# Print the dict formatted for direct copy-paste into the notebook hyperparams block.
# Also includes fixed parameters not tuned by Optuna.
print()
print("--- Copy-paste block for notebook hyperparams dict ---")
print(f'    "{GROUP}": dict(')
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
    "group":               GROUP,
    "best_binary_f1":      best_f1,
    "best_trial_number":   study.best_trial.number,
    "n_trials_completed":  len(study.trials),
    "n_folds":             10,
    "max_depth_ceiling":   MAX_DEPTH_CEILING,
    "use_location":        USE_LOCATION,
    "n_features":          len(feature_names),
    "best_params":         best_params,
    # Fixed params not tuned by Optuna — included for completeness
    "fixed_params": {
        "eval_metric":           "logloss",
        "early_stopping_rounds": 30,
        "random_state":          42,
    },
}

out_path = os.path.join(OUTPUT_DIR, f"optuna_group{GROUP}_results.json")
with open(out_path, "w") as f:
    json.dump(output_record, f, indent=4)

print(f"\nFull results saved to: {out_path}")