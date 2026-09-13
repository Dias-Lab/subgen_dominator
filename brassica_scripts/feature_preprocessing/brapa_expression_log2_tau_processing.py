"""
brapa_expression_log2_tau_processing.py
========================================
Preprocesses raw FPKM expression data for Brassica rapa to produce two
per-gene expression features used in the subgenome dominance ML model:

    1. Log2_Average  — Expression magnitude. The arithmetic mean of
                       log2(FPKM + 1) values across all 8 raw tissue
                       columns (Callus, Flower, Leaf1, Leaf2, Root1,
                       Root2, Silique, Stem).

    2. Tau_Index     — Expression breadth (tissue specificity). Calculated
                       from 6 biologically distinct tissue conditions after
                       averaging biological replicates:
                           Leaf  = mean(Leaf1, Leaf2)
                           Root  = mean(Root1, Root2)
                       This gives N=6 distinct tissues: Callus, Flower,
                       Leaf, Root, Silique, Stem.

Input
-----
GSE43245_genes.fpkm_tracking.txt
    Tab-separated Cufflinks FPKM tracking file from GEO accession GSE43245
    (Bai et al. 2014, BMC Genomics). Contains 8 tissue FPKM estimates
    alongside confidence intervals, mapped to the B. rapa v1.5 genome.
    Two metadata header rows precede the column header row.

    Tissue-to-column mapping (from file header comment row):
        q0_FPKM  ->  Callus
        q1_FPKM  ->  Flower
        q2_FPKM  ->  Leaf1   }  biological replicates,
        q7_FPKM  ->  Leaf2   }  averaged for Tau
        q3_FPKM  ->  Root1   }  biological replicates,
        q4_FPKM  ->  Root2   }  averaged for Tau
        q5_FPKM  ->  Silique
        q6_FPKM  ->  Stem

    Confidence interval columns (q*_conf_lo, q*_conf_hi) are intentionally
    excluded — only best-estimate FPKM values are used.

Output
------
Expression_Features_LogTau_Brapa.csv
    Three-column CSV: gene_id, Log2_Average, Tau_Index.
    Genes with FPKM = 0 across all 8 tissues receive NaN for Tau_Index.
    This is biologically meaningful (no expressed tissue detected) and is
    retained in the output. The main preprocessing notebook handles these
    NaNs explicitly.

Design notes
------------
Log2_Average uses all 8 raw tissue columns before replicate averaging.
This treats both root measurements and both leaf measurements as
independent observations, maximising the information used in the
magnitude estimate and matching the approach used for the maize model.

Tau_Index uses 6 biologically distinct tissue conditions after averaging
replicates. Using N=8 (with duplicate tissues) would systematically bias
Tau downward for tissue-specific genes, because a gene highly expressed
in root would score in 2 of 8 conditions rather than 1 of 6. Using N=6
correctly reflects the number of distinct tissue environments sampled.

IMPORTANT — cross-species comparability:
    The maize Tau_Index was calculated with N=24 tissues. The brassica
    Tau_Index uses N=6. Tau values are not numerically comparable between
    the maize and brassica models. Both indices correctly capture tissue
    specificity within their respective species, but should not be
    interpreted on the same scale. This is documented here and in the
    main preprocessing notebook.

Reference
---------
Bai et al. (2014) Transcriptome comparison of Brassica rapa in three
morphotypes: turnip, Chinese cabbage and oilseed. BMC Genomics 14:689.
GEO accession: GSE43245.
"""

import sys
import datetime
import pandas as pd
import numpy as np


# =============================================================================
# LOGGING UTILITY
# =============================================================================
 
class Tee:
    """
    Duplicates all write calls to two output streams simultaneously.
    See brapa_gc_content_processing.py for full usage notes.
    """
    def __init__(self, *streams):
        self.streams = streams
 
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
 
    def flush(self):
        for stream in self.streams:
            stream.flush()
            
            
def process_brapa_expression_features(input_path, output_path):
    """
    Process raw B. rapa FPKM data into Log2_Average and Tau_Index features.

    Parameters
    ----------
    input_path : str
        Path to the raw GSE43245 FPKM tracking file.
    output_path : str
        Path where the output CSV will be written.

    Returns
    -------
    pd.DataFrame
        Three-column dataframe: gene_id, Log2_Average, Tau_Index.
    """

    # -------------------------------------------------------------------------
    # BLOCK 1: Load raw data
    # -------------------------------------------------------------------------
    # The file has two metadata rows before the actual column header:
    #   Row 0: plain-text tissue description comment
    #   Row 1: blank
    #   Row 2: column headers (tracking_id, class_code, ..., q0_FPKM, ...)
    # skiprows=2 correctly positions the reader at the column header row.

    print("Loading raw FPKM data...")
    df = pd.read_csv(input_path, sep='\t', skiprows=2)
    print(f"  Raw file shape: {df.shape}")
    print(f"  Genes loaded: {len(df)}")

    # Rename gene ID column for consistency with downstream notebook
    df = df.rename(columns={'tracking_id': 'gene_id'})

    # Hard check: gene IDs must be unique — duplicates indicate upstream error
    if df['gene_id'].duplicated().any():
        n_dups = df['gene_id'].duplicated().sum()
        dup_ids = df.loc[df['gene_id'].duplicated(), 'gene_id'].unique().tolist()
        raise ValueError(
            f"CRITICAL: {n_dups} duplicate gene IDs detected in input file. "
            f"This must be resolved before proceeding.\n"
            f"Affected IDs (first 10): {dup_ids[:10]}"
        )
    print("  ✓ No duplicate gene IDs detected")

    # -------------------------------------------------------------------------
    # BLOCK 2: Select FPKM columns explicitly
    # -------------------------------------------------------------------------
    # Confidence interval columns (q*_conf_lo, q*_conf_hi) are excluded.
    # Only the 8 best-estimate FPKM columns are retained.
    # Explicit selection by suffix is used rather than positional slicing
    # to be robust to any column reordering in the source file.

    fpkm_cols = [c for c in df.columns if c.endswith('_FPKM')]

    expected_fpkm_cols = [
        'q0_FPKM', 'q1_FPKM', 'q2_FPKM', 'q3_FPKM',
        'q4_FPKM', 'q5_FPKM', 'q6_FPKM', 'q7_FPKM'
    ]
    if set(fpkm_cols) != set(expected_fpkm_cols):
        raise ValueError(
            f"CRITICAL: Unexpected FPKM columns detected.\n"
            f"Expected: {expected_fpkm_cols}\n"
            f"Found:    {fpkm_cols}"
        )
    print(f"  ✓ {len(fpkm_cols)} FPKM columns verified: {fpkm_cols}")

    # Validate no NaN values exist in the raw FPKM columns
    nan_counts = df[fpkm_cols].isna().sum()
    if nan_counts.any():
        raise ValueError(
            f"CRITICAL: NaN values detected in raw FPKM columns:\n{nan_counts[nan_counts > 0]}"
        )
    print("  ✓ No NaN values in raw FPKM columns")

    # -------------------------------------------------------------------------
    # BLOCK 3: Log2(FPKM + 1) transformation
    # -------------------------------------------------------------------------
    # The +1 pseudocount prevents log(0) for unexpressed genes while
    # compressing the dynamic range of the distribution.
    # Transformation is applied to all 8 columns simultaneously.

    raw_fpkm = df[fpkm_cols]
    log2_all8 = np.log2(raw_fpkm + 1)

    print(f"\n  Log2(FPKM+1) value range: "
          f"{log2_all8.values.min():.4f} to {log2_all8.values.max():.4f}")

    # -------------------------------------------------------------------------
    # BLOCK 4: Calculate Log2_Average (expression magnitude)
    # -------------------------------------------------------------------------
    # Arithmetic mean across all 8 log2-transformed tissue columns.
    # Both leaf replicates and both root replicates contribute equally
    # to this estimate, maximising use of available data.

    df['Log2_Average'] = log2_all8.mean(axis=1)

    print(f"  Log2_Average range: "
          f"{df['Log2_Average'].min():.4f} to {df['Log2_Average'].max():.4f}")

    # -------------------------------------------------------------------------
    # BLOCK 5: Average biological replicates for Tau calculation
    # -------------------------------------------------------------------------
    # Leaf1 (q2) and Leaf2 (q7) are biological replicates of leaf tissue.
    # Root1 (q3) and Root2 (q4) are biological replicates of root tissue.
    # Averaging them before Tau calculation gives N=6 biologically distinct
    # tissue conditions, preventing systematic downward bias in Tau for
    # tissue-specific genes.
    #
    # The 6 tissue conditions used for Tau:
    #   Callus  = q0_FPKM (log2-transformed)
    #   Flower  = q1_FPKM (log2-transformed)
    #   Leaf    = mean(q2_FPKM, q7_FPKM) (log2-transformed, then averaged)
    #   Root    = mean(q3_FPKM, q4_FPKM) (log2-transformed, then averaged)
    #   Silique = q5_FPKM (log2-transformed)
    #   Stem    = q6_FPKM (log2-transformed)

    log2_6tissue = pd.DataFrame({
        'Callus':  log2_all8['q0_FPKM'],
        'Flower':  log2_all8['q1_FPKM'],
        'Leaf':    log2_all8[['q2_FPKM', 'q7_FPKM']].mean(axis=1),
        'Root':    log2_all8[['q3_FPKM', 'q4_FPKM']].mean(axis=1),
        'Silique': log2_all8['q5_FPKM'],
        'Stem':    log2_all8['q6_FPKM'],
    })

    N = len(log2_6tissue.columns)  # N = 6
    print(f"\n  Tau calculation using N={N} distinct tissue conditions: "
          f"{log2_6tissue.columns.tolist()}")

    # -------------------------------------------------------------------------
    # BLOCK 6: Calculate Tau_Index (expression breadth / tissue specificity)
    # -------------------------------------------------------------------------
    # Tau formula: sum(1 - (x_i / x_max)) / (N - 1)
    #   x_i    = log2-transformed expression in tissue i
    #   x_max  = maximum log2-transformed expression across all N tissues
    #
    # Tau = 0: gene expressed equally across all tissues (housekeeping)
    # Tau = 1: gene expressed exclusively in a single tissue
    #
    # BIOLOGICAL EDGE CASE: Genes with FPKM = 0 across all 8 tissues have
    # x_max = 0 after log2 transformation. Division by zero would produce
    # undefined Tau. These genes are biologically real — they may be
    # developmental stage-specific or require conditions not sampled.
    # Their Tau is set to NaN and retained in the output. The main
    # preprocessing notebook handles these NaNs explicitly.

    max_expr = log2_6tissue.max(axis=1)
    expressed_mask = max_expr > 0

    n_unexpressed = (~expressed_mask).sum()
    print(f"\n  Genes with FPKM = 0 across all 8 tissues: {n_unexpressed}")
    print(f"  These genes receive NaN for Tau_Index (biologically meaningful).")

    df['Tau_Index'] = np.nan

    # Normalise each tissue by the gene's maximum, then apply Tau formula
    normalised = log2_6tissue[expressed_mask].div(max_expr[expressed_mask], axis=0)
    tau_values = (1 - normalised).sum(axis=1) / (N - 1)
    df.loc[expressed_mask, 'Tau_Index'] = tau_values

    print(f"  Tau_Index range (expressed genes): "
          f"{tau_values.min():.4f} to {tau_values.max():.4f}")

    # Sanity check: Tau must be in [0, 1] for all expressed genes
    out_of_bounds = ((tau_values < 0) | (tau_values > 1)).sum()
    if out_of_bounds > 0:
        raise ValueError(
            f"CRITICAL: {out_of_bounds} Tau values fall outside [0, 1]. "
            f"Review input data for anomalies."
        )
    print("  ✓ All Tau values are within [0, 1]")

    # -------------------------------------------------------------------------
    # BLOCK 7: Export clean feature matrix
    # -------------------------------------------------------------------------

    final_features = df[['gene_id', 'Log2_Average', 'Tau_Index']]

    print(f"\nSaving processed features to: {output_path}")
    final_features.to_csv(output_path, index=False)

    print(f"\n=== Processing complete ===")
    print(f"  Total genes processed  : {len(df)}")
    print(f"  Genes with Tau value   : {expressed_mask.sum()}")
    print(f"  Genes with NaN Tau     : {n_unexpressed}")
    print(f"  Output columns         : gene_id, Log2_Average, Tau_Index")

    return final_features

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python brapa_expression_log2_tau_processing.py <input_fpkm.txt> <output_features.csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = f'brapa_expression_log2_tau_{timestamp}.log'

    original_stdout = sys.stdout
    with open(log_path, 'w') as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        try:
            print(f"brapa_expression_log2_tau_processing.py")
            print(f"Run started : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Log file    : {log_path}\n")
            process_brapa_expression_features(
                input_path=input_path,
                output_path=output_path
            )
            print(f"\nRun completed : "
                  f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        finally:
            sys.stdout = original_stdout

    print(f"Log saved to: {log_path}")