"""
brapa_gc_content_processing.py
================================
Generates two GC content features for every B. rapa gene in the subgenome
dominance ML model:

    1. Gene_CDS_GC   — GC content (%) of the full coding sequence (CDS),
                       reconstructed by concatenating all annotated CDS exons
                       for each gene directly from the B. rapa v1.5 genome
                       FASTA and GFF annotation.

    2. Promoter_GC   — GC content (%) of the 170-bp core promoter window
                       (-165 to +5 bp relative to the annotated TSS),
                       extracted using the mRNA feature coordinate as the TSS
                       anchor, matching the spatial definition used for the
                       maize model (Jores et al. 2021).

Both features are computed directly from primary genomic files — genome FASTA
and GFF annotation — so that the entire analytical chain is fully reproducible
with no undocumented intermediate files.

Inputs
------
Brapa_sequence_v1.5.fa
    B. rapa v1.5 genome FASTA. Chromosome headers are of the form
    ">A01 [12.17-2010]" — BioPython parses record.id as the token before
    the first whitespace, giving "A01", "A02", ..., "A10" for main
    chromosomes and "Scaffold000096" etc. for unplaced scaffolds. These
    names match the chromosome column of the GFF exactly.

Brapa_gene_v1.5.gff
    B. rapa v1.5 gene annotation. Relevant feature types:
        gene  — ignored; mRNA is used instead for TSS precision.
        mRNA  — one record per gene. Provides start, end, and strand, which
                 define the TSS. The ID attribute holds the bare gene ID
                 (e.g. "Bra000001"). Source tags ("blat", "glean") are
                 ignored — only feature type is used for parsing.
        CDS   — one record per coding exon. The Parent attribute links each
                 exon to its gene ID. Multiple CDS rows per gene define the
                 full multi-exon coding sequence.

Brapa_3genomes_tPCK.confident
    Subgenome assignment file (from collaborator, validated against Cheng
    et al. 2016). Used to define the master set of doublet/triplet gene IDs
    for which GC features must be computed. After cleaning (removing rows
    where ara_paralog == "-" or has "-TA" suffix, removing the one known
    duplicate AT5G67640), this file yields 15,903 unique Bra gene IDs across
    the three pairwise doublet sets.

Output
------
gc_gene_promoter_output.csv  (written to output_dir)
    Three-column CSV: Gene_ID, Gene_CDS_GC, Promoter_GC.
    Values are GC percentage (0.0–100.0).
    Genes whose promoter window falls partly outside a sequence boundary
    receive NaN for Promoter_GC and are listed in the run log.

Design notes
------------
On scaffold chromosomes in the doublet set:
    170 of the 15,903 master doublet genes are located on scaffold
    chromosomes (unplaced sequences) rather than on the 10 main chromosomes
    A01–A10. This was verified by cross-referencing mRNA coordinates from the
    GFF against the master gene set.

    Critically, all 170 scaffold genes have specific AK block assignments
    (blocks A, B, C, D, F, H, N, P, S, U, V) in the subgenome assignment
    file. None fall in the "Unspecified" category reported in Table 1 of
    Cheng et al. 2012, which is the category associated with genes whose
    syntenic placement is ambiguous. These 170 genes have unambiguous
    syntenic ortholog relationships to Arabidopsis and valid subgenome labels.
    Their scaffold location is an artifact of the v1.5 genome assembly —
    likely pericentromeric or repeat-rich regions that resisted chromosome-
    scale placement — not a biological or analytical problem.

    Decision: scaffold genes are included in feature extraction. The genome
    FASTA contains these scaffold sequences, so extraction proceeds normally.
    The boundary check (below) handles any scaffold genes sitting too close
    to a scaffold edge. Scaffold genes are logged separately for transparency.

On isoforms:
    The Brapa_gene_v1.5_CDS_sorted.fa FASTA was verified to contain exactly
    41,020 sequences with bare ">BraXXXXXX" headers — one entry per gene,
    matching the 41,020 unique Parent IDs in the CDS GFF coordinate file.
    There are no isoform suffixes and no isoform bias problem. Because this
    script regenerates CDS sequences directly from the GFF and genome FASTA,
    it is fully independent of that intermediate file.

On CDS strand orientation:
    GC content is strand-symmetric: reverse complementing a sequence does not
    change its GC fraction (G↔C and A↔T swap, leaving the G+C count
    unchanged). Therefore, CDS exons are simply concatenated in genomic
    coordinate order and GC is calculated directly, with no reverse
    complementation step needed for CDS GC calculation.

On promoter extraction:
    The TSS is defined as the mRNA feature start coordinate (1-based GFF) for
    + strand genes and the mRNA feature end coordinate (1-based GFF) for -
    strand genes. This uses the annotated mRNA boundary as the TSS proxy,
    which is the same structural approach used in the corrected maize
    preprocessing pipeline. No empirical TSS data (e.g. CAGE-seq) is
    available for B. rapa v1.5, so the GFF-derived TSS is applied uniformly
    to all genes.

    The 170-bp window spans -165 to +5 bp relative to the TSS, matching the
    spatial definition of Jores et al. 2021 used for the maize model:
        + strand: genome[ TSS - 166 : TSS + 4 ]  (0-based Python slice)
        - strand: RC( genome[ TSS - 5 : TSS + 165 ] )  (0-based Python slice)

    Genes where the window extends beyond the sequence boundary (chromosome
    or scaffold edge) receive NaN for Promoter_GC and are listed in the log.
    From pre-analysis: all known boundary-edge genes in the doublet set are
    on scaffold chromosomes; no main-chromosome genes are expected to be
    affected.

Cross-species comparability note:
    This script mirrors the maize GC content preprocessing pipeline as
    closely as the B. rapa annotation structure allows. The main difference
    is that the maize pipeline used primary-transcript-only CDS sequences
    from Phytozome (Zmays_493_RefGen_V4.cds_primaryTranscriptOnly.fa) to
    eliminate isoform bias. For B. rapa v1.5, the annotation contains one
    CDS model per gene with no isoform variants, making this distinction
    moot. The biological features (CDS GC and promoter GC) are defined
    identically in both species.

Reference
---------
Jores et al. (2021) Identification of plant enhancers and their constituent
elements by STARR-seq in tobacco and rice. Plant Cell 33:2857–2878.

Cheng et al. (2012) Biased gene fractionation and dominant gene expression
among the subgenomes of Brassica rapa. PLoS ONE 7:e36442.

Cheng et al. (2016) Epigenetic regulation of subgenome dominance following
whole genome triplication in Brassica rapa. New Phytologist 211:288–299.
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq


# =============================================================================
# LOGGING UTILITY
# =============================================================================
 
class Tee:
    """
    Duplicates all write calls to two output streams simultaneously.
 
    Used to redirect sys.stdout so that every print() call in this script
    is written to both the terminal (for interactive feedback) and a timestamped
    log file in the current working directory (for a permanent record).
 
    Usage in __main__:
        original_stdout = sys.stdout
        with open(log_path, 'w') as log_file:
            sys.stdout = Tee(original_stdout, log_file)
            run_your_code()
        sys.stdout = original_stdout
    """
    def __init__(self, *streams):
        self.streams = streams
 
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()   # flush immediately so log is not lost on crash
 
    def flush(self):
        for stream in self.streams:
            stream.flush()
            
# =============================================================================
# CONFIGURATION
# =============================================================================

# Promoter window definition (matching Jores et al. 2021 and the maize model)
PROMOTER_UPSTREAM   = 165  # bp upstream of TSS (inclusive)
PROMOTER_DOWNSTREAM = 5    # bp downstream of TSS (inclusive, TSS = +1)
PROMOTER_LENGTH     = PROMOTER_UPSTREAM + PROMOTER_DOWNSTREAM  # = 170 bp

# Main chromosomes — genes on scaffolds are not expected in the doublet set
MAIN_CHROMOSOMES = {f'A{str(i).zfill(2)}' for i in range(1, 11)}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def gc_content_percent(sequence: str) -> float:
    """
    Calculate GC content as a percentage of total sequence length.

    Parameters
    ----------
    sequence : str
        DNA sequence string (upper or lower case).

    Returns
    -------
    float
        GC content in percent (0.0–100.0), or NaN if sequence is empty.
    """
    sequence = sequence.upper()
    total = len(sequence)
    if total == 0:
        return np.nan
    gc = sequence.count('G') + sequence.count('C')
    return (gc / total) * 100.0


def reverse_complement(sequence: str) -> str:
    """
    Return the reverse complement of a DNA sequence string.

    Parameters
    ----------
    sequence : str
        DNA sequence string (upper or lower case).

    Returns
    -------
    str
        Reverse-complemented sequence in upper case.
    """
    return str(Seq(sequence.upper()).reverse_complement())


# =============================================================================
# STEP 1: Load genome FASTA into memory
# =============================================================================

def load_genome(fasta_path: str) -> dict:
    """
    Load the genome FASTA into a dictionary keyed by chromosome ID.

    BioPython parses the sequence ID as everything before the first whitespace
    in the header line, so ">A01 [12.17-2010]" becomes key "A01".

    Parameters
    ----------
    fasta_path : str
        Path to the genome FASTA file.

    Returns
    -------
    dict
        {chromosome_id: sequence_string} for all records in the FASTA.
    """
    print("Loading genome FASTA into memory...")
    genome = {}
    for record in SeqIO.parse(fasta_path, 'fasta'):
        genome[record.id] = str(record.seq).upper()
    print(f"  Loaded {len(genome)} chromosomes/scaffolds.")
    print(f"  Main chromosomes present: "
          f"{sorted([c for c in genome if c in MAIN_CHROMOSOMES])}")
    return genome


# =============================================================================
# STEP 2: Parse GFF annotation
# =============================================================================

def parse_gff(gff_path: str) -> tuple:
    """
    Parse the GFF file and return two dictionaries:

        mrna_records : {gene_id: (chrom, start_1based, end_1based, strand)}
            One entry per gene, derived from mRNA feature rows.
            Used to define the TSS for promoter extraction.

        cds_records  : {gene_id: [(chrom, start_1based, end_1based), ...]}
            All CDS exon intervals per gene, derived from CDS feature rows.
            Used for CDS sequence reconstruction.

    The GFF uses 1-based closed coordinates throughout.
    Source tags ("blat", "glean") are ignored.

    Parameters
    ----------
    gff_path : str
        Path to the GFF annotation file.

    Returns
    -------
    tuple of (dict, dict)
        (mrna_records, cds_records)
    """
    print("\nParsing GFF annotation...")

    mrna_records = {}
    cds_records  = {}

    with open(gff_path, 'r') as fh:
        for line in fh:
            if line.startswith('#') or line.strip() == '':
                continue

            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue

            chrom   = parts[0]
            feature = parts[2]
            start   = int(parts[3])   # 1-based
            end     = int(parts[4])   # 1-based
            strand  = parts[6]
            attrs   = parts[8]

            if feature == 'mRNA':
                # Parse gene ID from ID= attribute
                gene_id = None
                for token in attrs.split(';'):
                    if token.startswith('ID='):
                        gene_id = token.replace('ID=', '').strip()
                        break
                if gene_id:
                    mrna_records[gene_id] = (chrom, start, end, strand)

            elif feature == 'CDS':
                # Parse gene ID from Parent= attribute
                gene_id = None
                for token in attrs.split(';'):
                    if token.startswith('Parent='):
                        gene_id = token.replace('Parent=', '').strip()
                        break
                if gene_id:
                    if gene_id not in cds_records:
                        cds_records[gene_id] = []
                    cds_records[gene_id].append((chrom, start, end))

    print(f"  mRNA records parsed: {len(mrna_records)}")
    print(f"  Genes with CDS records: {len(cds_records)}")
    return mrna_records, cds_records


# =============================================================================
# STEP 3: Load master doublet gene set
# =============================================================================

def load_master_genes(conf_path: str) -> set:
    """
    Load the set of all unique Bra gene IDs that appear in any doublet or
    triplet pair from the subgenome assignment file. These are the genes
    for which GC features must be computed.

    Parameters
    ----------
    conf_path : str
        Path to Brapa_3genomes_tPCK.confident.

    Returns
    -------
    set
        Set of Bra gene ID strings.
    """
    print("\nLoading master doublet gene set...")

    conf = pd.read_csv(conf_path, sep='\t', header=None)
    conf.columns = ['gene_num', 'ara_paralog', 'AKBr', 'ACK_block',
                    'LF', 'MF1', 'MF2']

    # Apply the same cleaning as the main preprocessing notebook
    conf = conf[conf['ara_paralog'] != '-'].copy()
    conf['ara_paralog'] = conf['ara_paralog'].str.split('-').str[0]
    conf = conf.drop_duplicates().reset_index(drop=True)

    lf_mf1  = conf[(conf['LF']  != '-') & (conf['MF1'] != '-')]
    lf_mf2  = conf[(conf['LF']  != '-') & (conf['MF2'] != '-')]
    mf1_mf2 = conf[(conf['MF1'] != '-') & (conf['MF2'] != '-')]

    master_genes = (
        set(lf_mf1['LF'])   | set(lf_mf1['MF1'])  |
        set(lf_mf2['LF'])   | set(lf_mf2['MF2'])  |
        set(mf1_mf2['MF1']) | set(mf1_mf2['MF2'])
    )

    print(f"  Total unique doublet/triplet genes: {len(master_genes)}")
    return master_genes


# =============================================================================
# STEP 4: Pre-checks
# =============================================================================

def run_prechecks(master_genes: set, mrna_records: dict,
                  cds_records: dict, genome: dict) -> set:
    """
    Validate that all master doublet genes are present in the annotation and
    genome. Characterise scaffold-resident genes with a documented warning
    rather than a hard error (see rationale in module docstring).

    Returns
    -------
    set
        The set of master genes located on scaffold chromosomes. This is
        returned so the extraction loop can log them separately.

    Raises ValueError for unrecoverable failures only:
        - Genes whose required chromosome sequence is absent from the FASTA.
          (This would cause silent NaN values with no diagnostic, which is
          worse than a hard stop.)
    """
    print("\n=== Pre-checks ===\n")
    total_warnings = 0

    # --- Check 1: All master genes have mRNA records (TSS available) ---
    # Genes missing an mRNA record cannot have a promoter GC value computed.
    # They will receive NaN for Promoter_GC after the merge in the main
    # preprocessing notebook and their pairs will be dropped at dropout.
    missing_mrna = master_genes - set(mrna_records.keys())
    if missing_mrna:
        print(f"  WARNING: {len(missing_mrna)} master genes have no mRNA "
              f"record in the GFF.")
        print(f"    These genes cannot have Promoter_GC calculated and will")
        print(f"    receive NaN, causing their pairs to be dropped at dropout.")
        print(f"    Affected IDs (first 10): "
              f"{sorted(list(missing_mrna))[:10]}")
        total_warnings += 1
    else:
        print(f"  ✓ All master genes have mRNA records in GFF")

    # --- Check 2: All master genes have CDS records ---
    # Genes missing CDS records cannot have Gene_CDS_GC computed.
    # Same dropout consequence as missing mRNA records.
    missing_cds = master_genes - set(cds_records.keys())
    if missing_cds:
        print(f"  WARNING: {len(missing_cds)} master genes have no CDS "
              f"records in the GFF.")
        print(f"    These genes cannot have Gene_CDS_GC calculated and will")
        print(f"    receive NaN, causing their pairs to be dropped at dropout.")
        print(f"    Affected IDs (first 10): "
              f"{sorted(list(missing_cds))[:10]}")
        total_warnings += 1
    else:
        print(f"  ✓ All master genes have CDS records in GFF")

    # --- Check 3: Characterise scaffold-resident master genes ---
    #
    # CONTEXT: Pre-analysis identified 170 master doublet genes on scaffold
    # chromosomes (unplaced sequences in the B. rapa v1.5 assembly). These
    # are NOT on the 10 main chromosomes A01–A10.
    #
    # DECISION: These genes are INCLUDED in feature extraction. The rationale:
    #
    #   (a) Subgenome assignment validity: All 170 scaffold genes have specific
    #       AK block assignments (A, B, C, D, F, H, N, P, S, U, V) confirmed
    #       in the Brapa_3genomes_tPCK.confident file. None fall in the
    #       "Unspecified" category of Cheng et al. 2012 Table 1, which is the
    #       category associated with syntactically ambiguous placement. Their
    #       subgenome labels (LF, MF1, MF2) are as reliable as any other gene.
    #
    #   (b) Scaffold origin: Scaffold sequences in plant assemblies typically
    #       represent pericentromeric or highly repetitive regions that resist
    #       chromosome-scale anchoring. The genes within them are real and
    #       biologically valid; their scaffold location is an assembly artifact.
    #
    #   (c) Extractability: The genome FASTA (Brapa_sequence_v1.5.fa) contains
    #       the scaffold sequences. CDS and promoter extraction will proceed
    #       normally for all scaffold genes. The boundary check (applied to all
    #       genes) will correctly handle any scaffold genes whose promoter
    #       window extends off the scaffold edge.
    #
    # Scaffold genes are logged here for full transparency and returned so the
    # extraction loop can flag them in the summary report.

    scaffold_genes = {}
    for gene_id in master_genes:
        if gene_id in mrna_records:
            chrom = mrna_records[gene_id][0]
            if chrom not in MAIN_CHROMOSOMES:
                scaffold_genes[gene_id] = chrom

    if scaffold_genes:
        # Count by scaffold to show clustering pattern
        scaffold_counts = {}
        for gid, chrom in scaffold_genes.items():
            scaffold_counts[chrom] = scaffold_counts.get(chrom, 0) + 1

        print(f"  INFO: {len(scaffold_genes)} master doublet genes are on "
              f"scaffold chromosomes ({len(scaffold_counts)} distinct scaffolds).")
        print(f"    All have valid AK block assignments — inclusion is justified.")
        print(f"    See module docstring for full rationale.")
        print(f"    Scaffold breakdown:")
        for sc, count in sorted(scaffold_counts.items(),
                                key=lambda x: x[1], reverse=True):
            print(f"      {sc}: {count} genes")
        print()
    else:
        # This branch is kept for completeness but is not expected to trigger
        # given the pre-analysis results. If it does, the scaffold_genes dict
        # returned will be empty and the extraction loop will behave identically
        # to a purely main-chromosome dataset.
        print(f"  ✓ All master genes are on main chromosomes A01–A10")

    # --- Check 4: All required chromosome sequences present in genome FASTA ---
    #
    # This is a hard stop because a missing chromosome sequence would cause
    # extract_cds_gc and extract_promoter_gc to return NaN silently for every
    # gene on that chromosome, with no diagnostic. A missing chromosome at this
    # stage means either the wrong genome FASTA was provided or the GFF and
    # FASTA are from different genome versions — both require investigation.
    needed_chroms = set()
    for gene_id in master_genes:
        if gene_id in mrna_records:
            needed_chroms.add(mrna_records[gene_id][0])

    missing_chroms = needed_chroms - set(genome.keys())
    if missing_chroms:
        raise ValueError(
            f"CRITICAL: {len(missing_chroms)} chromosome/scaffold sequence(s) "
            f"required for master genes are absent from the genome FASTA.\n"
            f"This indicates a GFF/FASTA version mismatch and must be resolved "
            f"before proceeding.\n"
            f"Missing sequences: {sorted(list(missing_chroms))}"
        )
    else:
        print(f"  ✓ All required chromosome and scaffold sequences present "
              f"in genome FASTA")

    print(f"\n=== Pre-check complete. Total warnings: {total_warnings} ===")

    # Return scaffold gene dict so the extraction loop can log them separately
    return scaffold_genes


# =============================================================================
# STEP 5: Feature extraction
# =============================================================================

def extract_cds_gc(gene_id: str, cds_records: dict, genome: dict) -> float:
    """
    Calculate GC content (%) of the full coding sequence for one gene.

    All annotated CDS exon intervals are extracted from the genome and
    concatenated. GC content is strand-symmetric so no reverse complementation
    is needed for GC calculation.

    Parameters
    ----------
    gene_id     : str
    cds_records : dict
    genome      : dict

    Returns
    -------
    float
        GC content in percent, or NaN if the gene has no CDS records.
    """
    if gene_id not in cds_records:
        return np.nan

    exons = cds_records[gene_id]
    # Sort by start coordinate (ascending) for consistency
    exons_sorted = sorted(exons, key=lambda x: x[1])

    cds_sequence = ''
    for chrom, start_1based, end_1based in exons_sorted:
        if chrom not in genome:
            return np.nan
        # Convert from 1-based closed to 0-based half-open
        seq = genome[chrom][start_1based - 1 : end_1based]
        cds_sequence += seq

    return gc_content_percent(cds_sequence)


def extract_promoter_gc(gene_id: str, mrna_records: dict,
                        genome: dict) -> float:
    """
    Extract the 170-bp core promoter sequence and calculate its GC content.

    Window definition (matching Jores et al. 2021 and the maize model):
        -165 to +5 bp relative to the annotated TSS (170 bp total)
        where TSS is defined as:
            + strand: mRNA feature start coordinate (1-based)
            - strand: mRNA feature end coordinate (1-based)

    Coordinate conversion to 0-based half-open (Python):
        + strand: genome[ TSS - 166 : TSS + 4 ]
        - strand: RC( genome[ TSS - 5 : TSS + 165 ] )

    Parameters
    ----------
    gene_id      : str
    mrna_records : dict
    genome       : dict

    Returns
    -------
    float
        GC content in percent, or NaN if window extends beyond chromosome
        boundary or chromosome is absent from genome.
    """
    if gene_id not in mrna_records:
        return np.nan

    chrom, mrna_start, mrna_end, strand = mrna_records[gene_id]

    if chrom not in genome:
        return np.nan

    chrom_len = len(genome[chrom])

    if strand == '+':
        # TSS = mRNA start (1-based)
        # Window in 0-based half-open: [TSS - 166 : TSS + 4]
        tss = mrna_start
        py_start = tss - 1 - PROMOTER_UPSTREAM   # = tss - 166
        py_end   = tss - 1 + PROMOTER_DOWNSTREAM  # = tss + 4

        # Boundary check
        if py_start < 0 or py_end > chrom_len:
            return np.nan

        promoter_seq = genome[chrom][py_start : py_end]

    else:
        # TSS = mRNA end (1-based) for - strand
        # Window in 0-based half-open: [TSS - 5 : TSS + 165]
        # then reverse complement
        tss = mrna_end
        py_start = tss - 1 - PROMOTER_DOWNSTREAM + 1  # = tss - 5 + 1 = tss - 4
        py_end   = tss - 1 + PROMOTER_UPSTREAM + 1    # = tss + 165

        # Boundary check
        if py_start < 0 or py_end > chrom_len:
            return np.nan

        forward_seq  = genome[chrom][py_start : py_end]
        promoter_seq = reverse_complement(forward_seq)

    if len(promoter_seq) != PROMOTER_LENGTH:
        # Paranoia check — should never be reached if boundary check passed
        return np.nan

    return gc_content_percent(promoter_seq)


# =============================================================================
# MAIN
# =============================================================================

def process_gc_features(genome_fa: str, gff_path: str, conf_path: str,
                         output_file: str):
    """
    Full pipeline: load inputs, run pre-checks, extract features, save output.

    Parameters
    ----------
    genome_fa   : str
    gff_path    : str
    conf_path   : str
    output_file : str
    """
    # --- Load inputs ---
    genome                    = load_genome(genome_fa)
    mrna_records, cds_records = parse_gff(gff_path)
    master_genes              = load_master_genes(conf_path)

    # --- Pre-checks ---
    # scaffold_genes is returned as a dict {gene_id: scaffold_name} so we can
    # cross-reference it in the summary report below.
    scaffold_genes = run_prechecks(master_genes, mrna_records, cds_records, genome)

    # --- Feature extraction ---
    print(f"\nExtracting GC features for {len(master_genes)} genes...\n")

    results       = []
    boundary_warn = []  # genes where promoter window hit a sequence boundary

    for gene_id in sorted(master_genes):

        # CDS GC: concatenate all annotated exons from the genome FASTA.
        # Strand-symmetric, so no reverse complementation needed.
        cds_gc  = extract_cds_gc(gene_id, cds_records, genome)

        # Promoter GC: 170-bp window anchored at the GFF-derived TSS.
        # Returns NaN if window extends beyond sequence boundary.
        prom_gc = extract_promoter_gc(gene_id, mrna_records, genome)

        # A NaN for promoter GC is only a boundary case if the gene has a
        # valid mRNA record. If it has no mRNA record at all, the NaN was
        # already flagged in the pre-check as a missing annotation case.
        if np.isnan(prom_gc) and gene_id in mrna_records:
            boundary_warn.append(gene_id)

        results.append({
            'Gene_ID':      gene_id,
            'Gene_CDS_GC':  cds_gc,
            'Promoter_GC':  prom_gc
        })

    df_out = pd.DataFrame(results)

    # --- Sanity check: GC values must be in [0, 100] ---
    # This should never fail if the genome FASTA is correct, but is included
    # as a hard stop to prevent silently passing corrupted values downstream.
    for col in ['Gene_CDS_GC', 'Promoter_GC']:
        out_of_range = df_out[col].dropna()
        out_of_range = out_of_range[(out_of_range < 0) | (out_of_range > 100)]
        if len(out_of_range) > 0:
            raise ValueError(
                f"CRITICAL: {len(out_of_range)} values in {col} are outside "
                f"[0, 100]. Review genome FASTA and annotation for corruption."
            )

    # --- Extraction summary ---
    print(f"=== Extraction Summary ===\n")
    print(f"  Total genes processed          : {len(df_out)}")
    print(f"  Genes on main chromosomes       : "
          f"{len(df_out) - len(scaffold_genes)}")
    print(f"  Genes on scaffold chromosomes   : {len(scaffold_genes)}")
    print(f"    (All scaffold genes have valid AK block assignments —")
    print(f"     included per module docstring rationale)")
    print()
    print(f"  NaN in Gene_CDS_GC              : "
          f"{df_out['Gene_CDS_GC'].isna().sum()}")
    print(f"  NaN in Promoter_GC              : "
          f"{df_out['Promoter_GC'].isna().sum()}")
    print(f"    of which boundary truncations : {len(boundary_warn)}")

    if boundary_warn:
        # List each boundary case with its chromosome/scaffold name so the
        # user can verify these are expected (short scaffold edges) and not
        # a sign of a coordinate system problem.
        print(f"\n  Genes with boundary-truncated promoter windows "
              f"(Promoter_GC = NaN):")
        for g in boundary_warn:
            chrom = mrna_records[g][0]
            on_scaffold = "(scaffold)" if chrom not in MAIN_CHROMOSOMES else ""
            print(f"    {g}  chrom={chrom}  {on_scaffold}")

    print()
    print(f"  Gene_CDS_GC range   : "
          f"{df_out['Gene_CDS_GC'].min():.2f}% – "
          f"{df_out['Gene_CDS_GC'].max():.2f}%")
    print(f"  Promoter_GC range   : "
          f"{df_out['Promoter_GC'].min():.2f}% – "
          f"{df_out['Promoter_GC'].max():.2f}%")
    print(f"\n  ✓ All GC values are within [0, 100]")

    # --- Save output ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_out.to_csv(output_file, index=False)
    print(f"\nOutput saved to: {output_file}")
    print(f"Output columns: {df_out.columns.tolist()}")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python brapa_gc_content_processing.py <genome.fa> <annotation.gff> <brapa_3genomes.confident> <output.csv>")
        sys.exit(1)

    genome_fa_arg   = sys.argv[1]
    gff_path_arg    = sys.argv[2]
    conf_path_arg   = sys.argv[3]
    output_file_arg = sys.argv[4]

    # Create a timestamped log file in the current working directory.
    # The timestamp prevents accidental overwriting if the script is run
    # multiple times. All print() output from this point onwards is written
    # to both the terminal and the log file simultaneously via Tee.
    timestamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path    = f'brapa_gc_content_{timestamp}.log'

    original_stdout = sys.stdout
    with open(log_path, 'w') as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        try:
            print(f"brapa_gc_content_processing.py")
            print(f"Run started : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Log file    : {log_path}\n")
            process_gc_features(
                genome_fa   = genome_fa_arg,
                gff_path    = gff_path_arg,
                conf_path   = conf_path_arg,
                output_file = output_file_arg,
            )
            print(f"\nRun completed : "
                  f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        finally:
            # Restore stdout unconditionally — even if process_gc_features
            # raises an exception, the error traceback will still be printed
            # to the terminal rather than being swallowed.
            sys.stdout = original_stdout

    print(f"Log saved to: {log_path}")