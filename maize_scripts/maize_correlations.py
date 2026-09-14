"""
Spearman correlation analysis for maize subgenome dominance feature matrices.

USAGE
-----
    python maize_correlations.py <GROUP>

Valid GROUP values: "All", "I", "II", "III", "IV"

OUTPUTS (per group, all filenames include group identifier; written to
maize_outputs/corr_output/, kept local — regenerable from final_featuresets/
and not tracked in git)
-------
Text files:
    corr_{GROUP}_pairwise_r0.3.txt       — all pairs with |r| >= 0.3
    corr_{GROUP}_pairwise_r0.5.txt       — all pairs with |r| >= 0.5
    corr_{GROUP}_Re_vs_all_features.txt  — Re vs every other feature

Figures:
    corr_{GROUP}_heatmap.png             — clustered Spearman correlation heatmap
    corr_{GROUP}_Re_barplot.png          — Re correlation bar plot

METHODS SUMMARY
---------------
Spearman rank correlation coefficients are computed pairwise across all
features using complete pairwise deletion (NaN values excluded per pair).
Significance is assessed with two-sided p-values and Bonferroni correction
applied across all n*(n-1)/2 feature pairs for the given group, where n
is the number of features.

Heatmap clustering uses average linkage on a distance matrix defined as
1 - |r|, where r is the Spearman correlation coefficient. This metric
treats positively and negatively correlated feature pairs symmetrically,
grouping co-varying blocks regardless of direction.

USE_LOCATION (controlled by GROUP):
  - All: location feature included (n = 60 features including location)
  - Groups I-IV: location excluded (n = 59 features)
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

FEATURESETS_DIR = Path("../final_featuresets")
OUTPUT_DIR = Path("../maize_outputs/corr_output")

INPUT_PATHS = {
    "All": FEATURESETS_DIR / "processedMaize_final.csv",
    "I":   FEATURESETS_DIR / "processedMaize_groupI_final.csv",
    "II":  FEATURESETS_DIR / "processedMaize_groupII_final.csv",
    "III": FEATURESETS_DIR / "processedMaize_groupIII_final.csv",
    "IV":  FEATURESETS_DIR / "processedMaize_groupIV_final.csv",
}

if len(sys.argv) != 2 or sys.argv[1] not in INPUT_PATHS:
    print(f"Usage: python maize_correlations.py <GROUP>")
    print(f"Valid GROUP values: {list(INPUT_PATHS.keys())}")
    sys.exit(1)

GROUP = sys.argv[1]

# location is included only for the All model
USE_LOCATION = (GROUP == "All")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
input_path = INPUT_PATHS[GROUP]

print("=" * 60)
print(f"GROUP        : {GROUP}")
print(f"USE_LOCATION : {USE_LOCATION}")
print(f"Input        : {input_path}")
print(f"Output dir   : {OUTPUT_DIR}")
print("=" * 60)

# ============================================================
# FEATURE RENAME MAP
# ============================================================
# Single authoritative rename dict covering all raw column names
# to short display names used in manuscript figures. Matches
# feature_rename_dict in networks_maize.ipynb exactly.
# recomb -> Re is included here rather than as a separate step.

RENAME_MAP = {
    "recomb":               "Re",
    "location":             "Loc",
    "GC_genic":             "GCd",
    "GC_prom":              "GCp",
    "avg_expression":       "Exp",
    "tau":                  "\u03c4",       # τ
    "Ka":                   "Ka",
    "Ks":                   "Ks",
    "O":                    "\u03c9",       # ω
    "acr_Lup":              "ACRdu",
    "acr_Ldown":            "ACRdd",
    "acr_S":                "ACRs",
    "CHH_up_avg":           "M1",
    "CHH_down_avg":         "M2",
    "CHH_body_avg":         "M3",
    "CHG_up_avg":           "M4",
    "CHG_down_avg":         "M5",
    "CHG_body_avg":         "M6",
    "CG_up_avg":            "M7",
    "CG_down_avg":          "M8",
    "CG_body_avg":          "M9",
    "CHH_up_max":           "M10",
    "CHH_down_max":         "M11",
    "CHG_up_max":           "M12",
    "CHG_down_max":         "M13",
    "CG_up_max":            "M14",
    "CG_down_max":          "M15",
    "TEdist":               "TEdi",
    "TEdenseAvgUp":         "TEu1",
    "TEdenseAvgDown":       "TEd1",
    "TEdenseMaxUp":         "TEu2",
    "TEdenseMaxDown":       "TEd2",
    "TE1":                  "TE1",
    "TE2":                  "TE2",
    "TE3":                  "TE3",
    "TE4":                  "TE4",
    "H2AZ_down":            "Hd1",
    "H3K4me1_down":         "Hd2",
    "H3K4me3_down":         "Hd3",
    "H3K9ac_down":          "Hd4",
    "H3K27ac_down":         "Hd5",
    "H3K27me3_down":        "Hd6",
    "H3K36me3_down":        "Hd7",
    "H3K56ac_down":         "Hd8",
    "H2AZ_genebody":        "Hg1",
    "H3K4me1_genebody":     "Hg2",
    "H3K4me3_genebody":     "Hg3",
    "H3K9ac_genebody":      "Hg4",
    "H3K27ac_genebody":     "Hg5",
    "H3K27me3_genebody":    "Hg6",
    "H3K36me3_genebody":    "Hg7",
    "H3K56ac_genebody":     "Hg8",
    "H2AZ_up":              "Hu1",
    "H3K4me1_up":           "Hu2",
    "H3K4me3_up":           "Hu3",
    "H3K9ac_up":            "Hu4",
    "H3K27ac_up":           "Hu5",
    "H3K27me3_up":          "Hu6",
    "H3K36me3_up":          "Hu7",
    "H3K56ac_up":           "Hu8",
}

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")
df = pd.read_csv(input_path)

assert "WGD" in df.columns, "CRITICAL: 'WGD' column not found."

# Exclude target label and, for Groups I-IV, the location feature
exclude = {"WGD"} if USE_LOCATION else {"WGD", "location"}
feature_cols = [c for c in df.columns if c not in exclude]
df_feat = df[feature_cols].copy()

# Apply rename map — only renames columns that exist in this dataset
df_feat = df_feat.rename(columns=RENAME_MAP)

features     = df_feat.columns.tolist()
n_features   = len(features)
n_pairs      = n_features * (n_features - 1) // 2

print(f"Features     : {n_features}")
print(f"Pairs        : {n_pairs:,}")
print(f"Samples      : {len(df_feat):,}")

nan_counts = df_feat.isna().sum()
nan_counts = nan_counts[nan_counts > 0]
if len(nan_counts):
    print("NaN features (pairwise deletion applied per pair):")
    for col, n in nan_counts.items():
        print(f"  {col}: {n} NaNs")

# ============================================================
# SPEARMAN CORRELATION MATRIX
# ============================================================

print("\nComputing Spearman correlation matrix...")
corr_matrix = df_feat.corr(method="spearman")

# ============================================================
# P-VALUE MATRIX (pairwise, with pairwise deletion for NaNs)
# ============================================================

print("Computing p-values (pairwise deletion per pair)...")

# Initialise with ones — diagonal stays 1.0 (self-correlation trivially p=1)
pval_matrix = pd.DataFrame(
    np.ones((n_features, n_features)),
    index=features, columns=features
)

for i in range(n_features):
    for j in range(i + 1, n_features):
        # dropna applied to each pair independently
        pair = df_feat[[features[i], features[j]]].dropna()
        _, p = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
        pval_matrix.iloc[i, j] = p
        pval_matrix.iloc[j, i] = p

# Bonferroni-corrected significance threshold across all n*(n-1)/2 pairs
alpha_bonferroni = 0.05 / n_pairs
print(f"Bonferroni threshold : p < {alpha_bonferroni:.2e}  ({n_pairs:,} pairs)")

# ============================================================
# TEXT OUTPUT 1: All pairs with |r| >= 0.3
# ============================================================

print("\nWriting pairwise correlation tables...")
rows = []
for i in range(n_features):
    for j in range(i + 1, n_features):
        r = corr_matrix.iloc[i, j]
        if abs(r) >= 0.3:
            rows.append({
                "Feature_1":             features[i],
                "Feature_2":             features[j],
                "Spearman_r":            round(r, 4),
                "p_value":               pval_matrix.iloc[i, j],
                "Significant_Bonferroni": pval_matrix.iloc[i, j] < alpha_bonferroni,
            })

df_03 = (pd.DataFrame(rows)
           .sort_values("Spearman_r", ascending=False)
           .reset_index(drop=True))
df_03["p_value"] = df_03["p_value"].map(lambda x: f"{x:.2e}")

path_03 = OUTPUT_DIR / f"corr_{GROUP}_pairwise_r0.3.txt"
df_03.to_csv(path_03, sep="\t", index=False)
print(f"  {len(df_03):>5} pairs |r|>=0.3  -> {path_03}")

# ============================================================
# TEXT OUTPUT 2: All pairs with |r| >= 0.5
# ============================================================

df_05 = df_03[df_03["Spearman_r"].abs() >= 0.5].copy()
path_05 = OUTPUT_DIR / f"corr_{GROUP}_pairwise_r0.5.txt"
df_05.to_csv(path_05, sep="\t", index=False)
print(f"  {len(df_05):>5} pairs |r|>=0.5  -> {path_05}")

# ============================================================
# TEXT OUTPUT 3: Re vs all features
# ============================================================

re_pvals = {
    col: pval_matrix.loc["Re", col]
    for col in features if col != "Re"
}
re_corr = corr_matrix["Re"].drop("Re")

re_df = pd.DataFrame({
    "Feature":               re_corr.index,
    "Spearman_r":            re_corr.values.round(4),
    "p_value":               [f"{re_pvals[f]:.2e}" for f in re_corr.index],
    "Significant_Bonferroni": [re_pvals[f] < alpha_bonferroni for f in re_corr.index],
})
# Sort by absolute correlation magnitude, descending
re_df = re_df.iloc[re_df["Spearman_r"].abs().argsort()[::-1]].reset_index(drop=True)

path_re = OUTPUT_DIR / f"corr_{GROUP}_Re_vs_all_features.txt"
re_df.to_csv(path_re, sep="\t", index=False)
print(f"  {len(re_df):>5} features         -> {path_re}")
print(f"\n  Re: {re_df['Significant_Bonferroni'].sum()} / {len(re_df)} features "
      f"significantly correlated (Bonferroni)")
print(f"  Max |r| with Re: {re_df['Spearman_r'].abs().max():.4f} "
      f"({re_df.iloc[0]['Feature']})")

# ============================================================
# FIGURE 1: CLUSTERED CORRELATION HEATMAP
# ============================================================
# Clustering uses average linkage on distance matrix 1 - |r|.
# This metric treats positively and negatively correlated pairs
# symmetrically, grouping co-varying feature blocks regardless
# of direction. The distance matrix diagonal is zeroed before
# converting to condensed form.
# ============================================================

print("\nPlotting clustered correlation heatmap...")

# Build distance matrix and linkage
# np.clip guards against floating-point values slightly outside [0, 1]
dist_matrix = np.clip(1.0 - corr_matrix.abs().values, 0.0, 1.0)
np.fill_diagonal(dist_matrix, 0.0)

condensed      = squareform(dist_matrix, checks=False)
row_linkage    = linkage(condensed, method="average")
col_linkage    = linkage(condensed, method="average")

cg = sns.clustermap(
    corr_matrix,
    row_linkage    = row_linkage,
    col_linkage    = col_linkage,
    cmap           = "RdBu_r",
    center         = 0,
    vmin           = -1,
    vmax           = 1,
    linewidths     = 0,           # no grid lines — cleaner at this feature count
    cbar_pos       = (0.01, 0.91, 0.02, 0.07),
    cbar_kws       = {"label": "Spearman \u03c1", "shrink": 0.5},
    dendrogram_ratio = 0.12,      # dendrograms take 12% of figure width/height
    figsize        = (22, 24),
    xticklabels    = True,
    yticklabels    = True,
)

# Rotate and size tick labels
cg.ax_heatmap.set_xticklabels(
    cg.ax_heatmap.get_xticklabels(), rotation=90, fontsize=7
)
cg.ax_heatmap.set_yticklabels(
    cg.ax_heatmap.get_yticklabels(), rotation=0, fontsize=7
)

# Bold the Re tick labels on both axes so it stands out
for label in cg.ax_heatmap.get_xticklabels():
    if label.get_text() == "Re":
        label.set_fontweight("bold")
        label.set_fontsize(8)
for label in cg.ax_heatmap.get_yticklabels():
    if label.get_text() == "Re":
        label.set_fontweight("bold")
        label.set_fontsize(8)

cg.ax_heatmap.set_xlabel("")
cg.ax_heatmap.set_ylabel("")

group_label = "All Maize1/Maize2" if GROUP == "All" else f"Group {GROUP}"
cg.figure.suptitle(
    f"Spearman correlation matrix — Maize {group_label}",
    fontsize=13, fontweight="bold", y=1.01
)

heatmap_path = OUTPUT_DIR / f"corr_{GROUP}_heatmap.png"
cg.figure.savefig(heatmap_path, dpi=600, bbox_inches="tight")
plt.close(cg.figure)
print(f"Saved: {heatmap_path}")

# ============================================================
# FIGURE 2: Re CORRELATION BAR PLOT
# ============================================================
# Features sorted by Spearman r with Re (ascending left to right
# on horizontal bars). Bonferroni-significant correlations marked *.
# ============================================================

print("Plotting Re correlation bar plot...")

# Sort ascending so most negative is at bottom, most positive at top
re_plot  = corr_matrix["Re"].drop("Re").sort_values()
bar_colors = ["#B22222" if v < 0 else "#4169E1" for v in re_plot.values]

fig, ax = plt.subplots(figsize=(14, 10))

ax.barh(range(len(re_plot)), re_plot.values,
        color=bar_colors, alpha=0.8, edgecolor="none")

# Mark Bonferroni-significant pairs with an asterisk
for i, feat in enumerate(re_plot.index):
    if re_pvals.get(feat, 1.0) < alpha_bonferroni:
        val = re_plot[feat]
        offset = 0.008 if val >= 0 else -0.008
        ax.text(val + offset, i, "*",
                ha="left" if val >= 0 else "right",
                va="center", fontsize=8, color="black")

ax.set_yticks(range(len(re_plot)))
ax.set_yticklabels(re_plot.index, fontsize=8)

ax.axvline(x=0,     color="black", linewidth=0.8)
ax.axvline(x=0.3,   color="grey",  linewidth=0.5, linestyle="--", alpha=0.5)
ax.axvline(x=-0.3,  color="grey",  linewidth=0.5, linestyle="--", alpha=0.5)

ax.set_xlabel("Spearman correlation coefficient", fontsize=11)
ax.set_title(
    f"Spearman correlation of recombination rate (Re)\n"
    f"with all maize {group_label} features",
    fontsize=13, fontweight="bold"
)
ax.set_xlim(-1, 1)

ax.text( 0.31, len(re_plot) * 1.02, "r = 0.3",
         color="grey", fontsize=8, ha="left")
ax.text(-0.31, len(re_plot) * 1.02, "r = \u22120.3",
         color="grey", fontsize=8, ha="right")

ax.legend(
    handles=[
        Patch(facecolor="#B22222", alpha=0.8, label="Negative correlation with Re"),
        Patch(facecolor="#4169E1", alpha=0.8, label="Positive correlation with Re"),
    ],
    loc="lower right", fontsize=9
)
ax.text(
    0.98, 0.02,
    f"* Bonferroni-significant (p < {alpha_bonferroni:.1e})",
    transform=ax.transAxes, ha="right", va="bottom",
    fontsize=7.5, color="gray", style="italic"
)

plt.tight_layout()

barplot_path = OUTPUT_DIR / f"corr_{GROUP}_Re_barplot.png"
plt.savefig(barplot_path, dpi=600, bbox_inches="tight")
plt.close()
print(f"Saved: {barplot_path}")

print("\nDone.")