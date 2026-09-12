"""
assign_ACR_features_maize_updated_dist.py

Purpose:
    For each B73 maize gene (chromosomes Chr1-Chr10), compute three features
    for use in an XGBoost model:

        1. summit_fold_enrichment  -- summit-height of the most relevant ACR
        2. upstream_distance       -- distance from gene to closest upstream ACR
        3. downstream_distance     -- distance from gene to closest downstream ACR

Key decisions & Methodological Updates:
    1.  The summit score is summit-height from the ATAC-seq BED file.
        This is the direct analog of fold_enrichment from the brassica MACS2 BED output.

    2.  Coordinate systems: GFF3 is 1-based; BED is 0-based. GFF3 start
        coordinates are converted to 0-based by subtracting 1 so that both
        files share the same coordinate space.

    3.  Chromosome name normalization: the GFF3 uses "Chr1"-"Chr10" while
        the BED uses bare "1"-"10". BED chromosome names are normalized to
        "Chr1"-"Chr10" format on load so the two files share the same keys.

    4.  Strand-aware TSS/TTS:
            + strand: TSS = gene start (lower coord), TTS = gene end (higher coord)
            - strand: TSS = gene end   (higher coord), TTS = gene start (lower coord)

    5.  ACR positional categories & Distance calculation (UPDATED):
            Overlapping (genic) ACRs are no longer assigned an artificial distance of 0.0.
            Instead, true absolute physical distances from the ACR summit to the nearest
            gene boundary (TSS or TTS) are calculated.
            They are categorized as 'upstream' or 'downstream' depending on which
            boundary they are physically closest to.

    6.  Fold Enrichment Selection (Decoupled & UPDATED):
            The logic for fold enrichment is strictly decoupled from distance:
            - If any ACR overlaps the gene, the script strictly takes the maximum
              summit score among the overlapping ACRs.
            - If no ACR overlaps the gene, the script selects the summit score of the
              absolute closest flanking ACR.

Usage:
    python assign_ACR_features_maize_updated_dist.py <genes.gff3> <acr_peaks.bed> <output.tsv>
"""

import sys
import pandas as pd
import numpy as np

# ===========================================================================
# 1. File paths
# ===========================================================================
if len(sys.argv) != 4:
    print("Usage: python assign_ACR_features_maize_updated_dist.py <genes.gff3> <acr_peaks.bed> <output.tsv>")
    sys.exit(1)

GFF3_FILE = sys.argv[1]  # Gene-only GFF3 (e.g., Zm-B73-REFERENCE-GRAMENE-4.0_Zm00001d.2.gff3_genes)
BED_FILE = sys.argv[2]   # ATAC-seq ACR peak calls (e.g., GSM3398046_ATAC_B73_leaf.filtered_ACR.bed)
OUT_FILE = sys.argv[3]   # Output path for merged ACR feature TSV

valid_chroms = {f"Chr{i}" for i in range(1, 11)}

# ---------------------------------------------------------------------------
# Load GFF3 (Genes only)
# ---------------------------------------------------------------------------
print("Loading GFF3...")
gff_cols = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
gff = pd.read_csv(GFF3_FILE, sep="\t", comment="#", header=None, names=gff_cols)

genes = gff[(gff["type"] == "gene") & (gff["seqid"].isin(valid_chroms))].copy()

# Extract gene_id from attributes
# Assuming format: ID=gene:Zm00001d027230;... or ID=Zm00001d027230;...
genes["gene_id"] = genes["attributes"].str.extract(r'ID=(?:gene:)?([^;]+)')

# Convert 1-based start to 0-based start
genes["start"] = genes["start"] - 1

# ---------------------------------------------------------------------------
# Load BED (ACRs)
# ---------------------------------------------------------------------------
print("Loading BED...")
bed_cols = ["chrom", "start", "end", "type", "abs_summit", "fold_enrichment"]
bed = pd.read_csv(BED_FILE, sep="\t", comment="#", header=None, names=bed_cols)

# Format chromosome names to match GFF3 ("1" -> "Chr1")
bed["chrom"] = "Chr" + bed["chrom"].astype(str)
bed = bed[bed["chrom"].isin(valid_chroms)].copy()

# ---------------------------------------------------------------------------
# Pre-group ACRs by chromosome for faster lookups
# ---------------------------------------------------------------------------
acr_by_chrom = {chrom: df for chrom, df in bed.groupby("chrom")}

# ---------------------------------------------------------------------------
# Calculate Features per Gene
# ---------------------------------------------------------------------------
print("Calculating features...")
results = []

for _, gene in genes.iterrows():
    chrom = gene["seqid"]
    gene_start = gene["start"]
    gene_end = gene["end"]
    strand = gene["strand"]
    gene_id = gene["gene_id"]

    if chrom not in acr_by_chrom:
        results.append({
            "gene_id": gene_id,
            "summit_fold_enrichment": np.nan,
            "upstream_distance": np.nan,
            "downstream_distance": np.nan,
        })
        continue

    acr_df = acr_by_chrom[chrom]

    acr_starts = acr_df["start"].values
    acr_ends = acr_df["end"].values
    summits = acr_df["abs_summit"].values
    fe = acr_df["fold_enrichment"].values

    if len(acr_df) == 0:
        summit_fe = np.nan
        upstream_dist = np.nan
        downstream_dist = np.nan
    else:
        distances = np.full(len(acr_df), np.inf)
        categories = np.array(["none"] * len(acr_df), dtype=object)

        # ---------------------------------------------------------
        # 1. Calculate Distances for Strict Flanking ACRs
        # ---------------------------------------------------------
        if strand == "+":
            left_mask = summits < gene_start
            if left_mask.any():
                distances[left_mask] = gene_start - summits[left_mask]
                categories[left_mask] = "upstream"

            right_mask = summits > gene_end
            if right_mask.any():
                distances[right_mask] = summits[right_mask] - gene_end
                categories[right_mask] = "downstream"
        else:  # strand == "-"
            left_mask = summits < gene_start
            if left_mask.any():
                distances[left_mask] = gene_start - summits[left_mask]
                categories[left_mask] = "downstream"

            right_mask = summits > gene_end
            if right_mask.any():
                distances[right_mask] = summits[right_mask] - gene_end
                categories[right_mask] = "upstream"

        # ---------------------------------------------------------
        # 2. Overlapping ACRs: True Physical Distance to Nearest Boundary
        # ---------------------------------------------------------
        overlap_mask = (acr_ends > gene_start) & (acr_starts < gene_end)

        if overlap_mask.any():
            tss = gene_start if strand == "+" else gene_end
            tts = gene_end if strand == "+" else gene_start

            dist_to_tss = np.abs(summits[overlap_mask] - tss)
            dist_to_tts = np.abs(summits[overlap_mask] - tts)

            distances[overlap_mask] = np.minimum(dist_to_tss, dist_to_tts)

            overlap_cats = np.where(dist_to_tss <= dist_to_tts, "upstream", "downstream")
            categories[overlap_mask] = overlap_cats

        # ---------------------------------------------------------
        # 3. Decoupled Fold Enrichment Selection
        # ---------------------------------------------------------
        if overlap_mask.any():
            summit_fe = fe[overlap_mask].max()
        else:
            valid_dist_mask = distances < np.inf
            if valid_dist_mask.any():
                best_idx = np.argmin(distances)
                summit_fe = fe[best_idx]
            else:
                summit_fe = np.nan

        # ---------------------------------------------------------
        # 4. Final Feature Extraction
        # ---------------------------------------------------------
        upstream_mask = categories == "upstream"
        upstream_dist = distances[upstream_mask].min() if upstream_mask.any() else np.nan

        downstream_mask = categories == "downstream"
        downstream_dist = distances[downstream_mask].min() if downstream_mask.any() else np.nan

    results.append({
        "gene_id": gene_id,
        "summit_fold_enrichment": summit_fe,
        "upstream_distance": upstream_dist,
        "downstream_distance": downstream_dist,
    })

# ---------------------------------------------------------------------------
# Build output dataframe
# ---------------------------------------------------------------------------
out_df = pd.DataFrame(results, columns=[
    "gene_id", "summit_fold_enrichment", "upstream_distance", "downstream_distance"
])

out_df.to_csv(OUT_FILE, sep="\t", index=False)

print(f"Total genes processed: {len(out_df)}")
print(f"Data saved to {OUT_FILE}")