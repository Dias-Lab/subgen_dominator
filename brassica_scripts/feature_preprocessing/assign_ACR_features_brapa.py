"""
assign_ACR_features_brapa.py

Purpose:
    For each B. rapa gene (chromosomes A01-A10), compute three ACR features
    for use in an XGBoost subgenome dominance model:

        1. summit_fold_enrichment  -- fold enrichment at the summit of the
                                      most relevant ACR near the gene
        2. upstream_distance       -- distance from gene to closest upstream ACR
        3. downstream_distance     -- distance from gene to closest downstream ACR

    Only genes on main chromosomes A01-A10 are processed. Genes on scaffold
    chromosomes receive NaN for all three features. This is consistent with
    the treatment of scaffold genes throughout the B. rapa preprocessing
    pipeline (scaffold genes also lack recombination rate data).

Input files:
    GFF_FILE : Brapa_gene_v1.5.gff
        B. rapa v1.5 gene annotation. Feature type 'gene' rows are used.
        GFF coordinates are 1-based closed intervals; start coordinates are
        converted to 0-based on load to match BED coordinate space.

    BED_FILE : ACRs_8_libraries_combined.bed
        ATAC-seq accessible chromatin regions (ACRs) from 8 combined
        libraries, processed with MACS2. Columns:
            chr, start, end, length, abs_summit, pileup,
            neg_log10_pval, fold_enrichment, neg_log10_qval, name
        BED coordinates are 0-based. Chromosome names match GFF directly
        (A01-A10) — no name normalization is needed, unlike the maize script.

Output:
    TSV file with columns: gene_id, summit_fold_enrichment,
                           upstream_distance, downstream_distance

Methodological notes:
    This script is adapted directly from assign_ACR_features_maize_updated_dist.py.
    The algorithm, strand-aware TSS/TTS logic, overlapping vs flanking
    classification, and decoupled fold enrichment selection are identical.
    Differences from the maize script:

    1.  Chromosome names: brassica BED already uses A01-A10, matching the GFF.
        No "Chr" + str normalization is needed.

    2.  BED format: the brassica BED has a header row; the maize BED did not.
        The header is parsed and columns are named on load.

    3.  Valid chromosomes: A01-A10 (not Chr1-Chr10 as in maize).

    4.  GFF feature type: 'gene' features are used (same as maize).

    5.  Fold enrichment values: the brassica ACR BED contains a small number
        of peaks with extreme fold enrichment values (max: 9195.3). These are
        real, high-pileup peaks that happen to fall in regions of near-zero
        background — they are not noise artifacts. Only 26 of 15,903 master
        doublet genes are within 5 kb of any peak with FE > 100, and only 1
        gene is within 5 kb of a peak with FE > 1000. Values are retained
        as-is and the distribution is documented here for transparency.
        The fold enrichment percentiles (across all 22,486 peaks) are:
            50th  :   4.96
            75th  :   6.14
            90th  :   8.38
            95th  :  11.24
            99th  :  40.15
            99.9th: 1182.08
            100th : 9195.30

    All other methodological decisions — coordinate system handling, strand
    aware TSS/TTS definition, overlapping ACR physical distance calculation,
    and decoupled fold enrichment selection — are identical to the maize
    script and are documented in the docstring of that script.

ACR NaN handling in the main preprocessing notebook:
    Genes that receive NaN for upstream_distance or downstream_distance
    because no ACR was found within the analysis window are biologically
    meaningful — they are genes with no nearby accessible chromatin. These
    NaN values are added to NAN_EXEMPT_COLS in the main preprocessing
    notebook and are retained rather than triggering pair dropout.
    Genes that receive NaN because they are on scaffold chromosomes (no ACR
    data available) also receive NaN, but this is handled the same way.
"""

import sys
import datetime
import os
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


# =============================================================================
# CONFIGURATION
# =============================================================================

# Main chromosomes only — scaffold genes receive NaN
VALID_CHROMS = {f'A{str(i).zfill(2)}' for i in range(1, 11)}


# =============================================================================
# LOAD GFF (gene features only)
# =============================================================================

def load_genes(gff_path: str) -> pd.DataFrame:
    """
    Load gene features from the B. rapa v1.5 GFF annotation.

    Only rows with feature type 'gene' on main chromosomes A01-A10 are kept.
    GFF coordinates are 1-based; start is converted to 0-based to match BED.

    Parameters
    ----------
    gff_path : str

    Returns
    -------
    pd.DataFrame
        Columns: gene_id, chrom, start (0-based), end, strand
    """
    print("Loading GFF annotation...")
    gff_cols = ['chrom', 'source', 'feature', 'start', 'end',
                'score', 'strand', 'phase', 'attrs']
    gff = pd.read_csv(gff_path, sep='\t', header=None,
                      comment='#', names=gff_cols)

    genes = gff[
        (gff['feature'] == 'gene') &
        (gff['chrom'].isin(VALID_CHROMS))
    ].copy()

    # Extract gene ID from the ID= attribute
    genes['gene_id'] = genes['attrs'].str.extract(r'ID=([^;]+)')

    # Convert 1-based GFF start to 0-based to match BED coordinate space
    genes['start'] = genes['start'] - 1

    genes = genes[['gene_id', 'chrom', 'start', 'end', 'strand']].copy()
    genes = genes.reset_index(drop=True)

    print(f"  Genes on main chromosomes A01-A10: {len(genes)}")

    # Hard check: no duplicate gene IDs
    dups = genes['gene_id'].duplicated()
    if dups.any():
        raise ValueError(
            f"CRITICAL: {dups.sum()} duplicate gene_ids in GFF gene features.\n"
            f"Affected IDs: {genes.loc[dups, 'gene_id'].tolist()[:10]}"
        )
    print("  ✓ No duplicate gene_ids in GFF")
    return genes


# =============================================================================
# LOAD BED (ACR peaks)
# =============================================================================

def load_acrs(bed_path: str) -> pd.DataFrame:
    """
    Load ATAC-seq ACR peaks from the combined 8-library BED file.

    The brassica BED has a header row and chromosome names already in A01-A10
    format, matching the GFF directly. No chromosome name normalization is
    needed (contrast with the maize script which required "Chr" + str).

    The two '#NAME?' columns are corrupted Excel formula artifacts
    (neg_log10_pval and neg_log10_qval from MACS2 output). They are renamed
    here for clarity but are not used in feature calculation.

    Parameters
    ----------
    bed_path : str

    Returns
    -------
    pd.DataFrame
        Filtered to VALID_CHROMS only.
    """
    print("\nLoading ACR BED file...")
    bed = pd.read_csv(bed_path, sep='\t', header=0)
    bed.columns = ['chr', 'start', 'end', 'length', 'abs_summit', 'pileup',
                   'neg_log10_pval', 'fold_enrichment', 'neg_log10_qval', 'name']

    n_total = len(bed)
    bed = bed[bed['chr'].isin(VALID_CHROMS)].copy().reset_index(drop=True)
    n_scaffold = n_total - len(bed)

    print(f"  Total peaks in file     : {n_total}")
    print(f"  Peaks on scaffold chroms: {n_scaffold} (excluded — no gene models)")
    print(f"  Peaks on A01-A10        : {len(bed)}")
    print(f"  fold_enrichment range   : "
          f"{bed['fold_enrichment'].min():.2f} to "
          f"{bed['fold_enrichment'].max():.2f}")

    # Note on extreme fold enrichment values — see module docstring
    n_extreme = (bed['fold_enrichment'] > 100).sum()
    if n_extreme > 0:
        print(f"\n  NOTE: {n_extreme} peaks with fold_enrichment > 100 on main chromosomes.")
        print(f"  These are real, high-pileup peaks in low-background regions.")
        print(f"  Values are retained as-is. Only 26 of 15,903 master doublet")
        print(f"  genes are within 5kb of any such peak. See module docstring.")

    return bed


# =============================================================================
# CALCULATE ACR FEATURES PER GENE
# =============================================================================

def calculate_acr_features(genes: pd.DataFrame,
                            bed: pd.DataFrame) -> pd.DataFrame:
    """
    For each gene, compute summit_fold_enrichment, upstream_distance, and
    downstream_distance using the same algorithm as the maize script.

    Algorithm summary:
        1.  For each ACR summit, calculate distance and assign direction
            (upstream / downstream) relative to the gene, using strand-aware
            TSS and TTS coordinates.
        2.  For overlapping ACRs (summit inside gene body): calculate true
            physical distance to the nearest gene boundary (TSS or TTS) and
            assign direction based on which boundary is closer. This avoids
            the artificial distance=0 that would result from treating all
            overlapping ACRs identically.
        3.  Fold enrichment is selected independently from distance:
            - If any ACR overlaps the gene, take the max fold enrichment
              among overlapping ACRs only.
            - If no ACR overlaps, take the fold enrichment of the single
              closest flanking ACR (by summit distance).
        4.  Upstream and downstream distances are the minimum distances
            among all ACRs assigned to each category.

    Parameters
    ----------
    genes : pd.DataFrame
    bed   : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        One row per gene with columns:
        gene_id, summit_fold_enrichment, upstream_distance, downstream_distance
    """
    print("\nCalculating ACR features...")

    # Pre-group ACRs by chromosome for fast lookup
    acr_by_chrom = {chrom: grp for chrom, grp in bed.groupby('chr')}

    results = []

    for _, gene in genes.iterrows():
        chrom    = gene['chrom']
        gene_s   = gene['start']    # 0-based
        gene_e   = gene['end']      # 0-based
        strand   = gene['strand']
        gene_id  = gene['gene_id']

        # Genes on chromosomes with no ACRs receive NaN
        if chrom not in acr_by_chrom or len(acr_by_chrom[chrom]) == 0:
            results.append({
                'gene_id':               gene_id,
                'summit_fold_enrichment': np.nan,
                'upstream_distance':      np.nan,
                'downstream_distance':    np.nan,
            })
            continue

        acr_df   = acr_by_chrom[chrom]
        summits  = acr_df['abs_summit'].values
        fe       = acr_df['fold_enrichment'].values
        acr_s    = acr_df['start'].values
        acr_e    = acr_df['end'].values

        # --- Step 1: Baseline distance calculation for all ACRs ---
        distances  = np.full(len(acr_df), np.inf)
        categories = np.array(['none'] * len(acr_df), dtype=object)

        if strand == '+':
            left_mask = summits < gene_s
            if left_mask.any():
                distances[left_mask]  = gene_s - summits[left_mask]
                categories[left_mask] = 'upstream'

            right_mask = summits > gene_e
            if right_mask.any():
                distances[right_mask]  = summits[right_mask] - gene_e
                categories[right_mask] = 'downstream'
        else:
            # - strand: left of gene is downstream, right is upstream
            left_mask = summits < gene_s
            if left_mask.any():
                distances[left_mask]  = gene_s - summits[left_mask]
                categories[left_mask] = 'downstream'

            right_mask = summits > gene_e
            if right_mask.any():
                distances[right_mask]  = summits[right_mask] - gene_e
                categories[right_mask] = 'upstream'

        # --- Step 2: Overlapping ACRs — true physical distance to nearest boundary ---
        overlap_mask = (acr_e > gene_s) & (acr_s < gene_e)

        if overlap_mask.any():
            # Strand-aware TSS and TTS
            tss = gene_s if strand == '+' else gene_e
            tts = gene_e if strand == '+' else gene_s

            dist_to_tss = np.abs(summits[overlap_mask] - tss)
            dist_to_tts = np.abs(summits[overlap_mask] - tts)

            distances[overlap_mask]  = np.minimum(dist_to_tss, dist_to_tts)
            categories[overlap_mask] = np.where(
                dist_to_tss <= dist_to_tts, 'upstream', 'downstream'
            )

        # --- Step 3: Decoupled fold enrichment selection ---
        if overlap_mask.any():
            # Overlapping ACRs: take the maximum fold enrichment among them
            summit_fe = fe[overlap_mask].max()
        else:
            valid_mask = distances < np.inf
            if valid_mask.any():
                best_idx  = np.argmin(distances)
                summit_fe = fe[best_idx]
            else:
                summit_fe = np.nan

        # --- Step 4: Final upstream and downstream minimum distances ---
        up_mask   = categories == 'upstream'
        down_mask = categories == 'downstream'

        upstream_dist   = distances[up_mask].min()   if up_mask.any()   else np.nan
        downstream_dist = distances[down_mask].min() if down_mask.any() else np.nan

        results.append({
            'gene_id':               gene_id,
            'summit_fold_enrichment': summit_fe,
            'upstream_distance':      upstream_dist,
            'downstream_distance':    downstream_dist,
        })

    out_df = pd.DataFrame(results, columns=[
        'gene_id', 'summit_fold_enrichment',
        'upstream_distance', 'downstream_distance'
    ])

    # --- Summary ---
    print(f"  Total genes processed         : {len(out_df)}")
    print(f"  Genes with NaN fold_enrichment: "
          f"{out_df['summit_fold_enrichment'].isna().sum()}")
    print(f"  Genes with NaN upstream_dist  : "
          f"{out_df['upstream_distance'].isna().sum()}")
    print(f"  Genes with NaN downstream_dist: "
          f"{out_df['downstream_distance'].isna().sum()}")
    print(f"\n  summit_fold_enrichment range  : "
          f"{out_df['summit_fold_enrichment'].min():.2f} to "
          f"{out_df['summit_fold_enrichment'].max():.2f}")

    return out_df


# =============================================================================
# MAIN
# =============================================================================

def main(gff_file, bed_file, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    genes  = load_genes(gff_file)
    bed    = load_acrs(bed_file)
    out_df = calculate_acr_features(genes, bed)

    out_df.to_csv(out_file, sep='\t', index=False)
    print(f"\nOutput saved to: {out_file}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python assign_ACR_features_brapa.py <genes.gff> <acr_peaks.bed> <output.tsv>")
        sys.exit(1)

    gff_file_arg = sys.argv[1]
    bed_file_arg = sys.argv[2]
    out_file_arg = sys.argv[3]

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = f'brapa_acr_features_{timestamp}.log'

    original_stdout = sys.stdout
    with open(log_path, 'w') as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        try:
            print("assign_ACR_features_brapa.py")
            print(f"Run started : "
                  f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Log file    : {log_path}\n")
            main(gff_file_arg, bed_file_arg, out_file_arg)
            print(f"\nRun completed : "
                  f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        finally:
            sys.stdout = original_stdout

    print(f"Log saved to: {log_path}")