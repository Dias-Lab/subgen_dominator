import sys
import csv

# Usage Check
if len(sys.argv) != 8:
    print("Usage: python gc_calc_prom_gene_processor.py <genome.fa> <annotation.gff3> <output_merged.csv> <cds_primary.fa> <mismatch_log.txt> <jores_sup.csv> <4578_duplicated_pairs_v4.csv>")
    sys.exit(1)

fasta_file = sys.argv[1]       # Genome assembly FASTA (e.g., B73 reference)
gff_file = sys.argv[2]         # GFF3 gene annotation
out_file = sys.argv[3]         # Output path for merged GC content CSV
cds_fasta_file = sys.argv[4]   # Phytozome primary-transcript-only CDS FASTA
log_file = sys.argv[5]         # Output path for mismatch log
jores_file = sys.argv[6]       # Jores et al. supplementary promoter GC dataset
pairs_file = sys.argv[7]       # Duplicated gene-pairs CSV (defines target gene set)

complement_map = str.maketrans('ACGTacgtNn', 'TGCAtgcaNn')

# -----------------------------------------------------------------------------
# BLOCK 0: Isolate Required Feature Space (Target Genes)
# -----------------------------------------------------------------------------
print("Loading strictly required genes from duplicate pairs matrix...")
required_genes = set()
with open(pairs_file, "r", encoding='utf-8-sig') as pf:
    reader = csv.DictReader(pf)
    for row in reader:
        required_genes.add(row['Maize1'])
        required_genes.add(row['Maize2'])
print(f"Isolated {len(required_genes)} required homoeologs.")

# -----------------------------------------------------------------------------
# BLOCK 1: Parse Phytozome CDS FASTA (Strictly for Required Genes)
# -----------------------------------------------------------------------------
print("Parsing Phytozome Primary Transcripts for target genes...")
primary_transcripts = {}
current_transcript_id = None
current_base_gene = None
current_seq = []

with open(cds_fasta_file, "r") as cds_in:
    for line in cds_in:
        line = line.strip()
        if line.startswith(">"):
            if current_transcript_id and current_seq and current_base_gene in required_genes:
                full_cds_seq = "".join(current_seq)
                gc_count = full_cds_seq.upper().count('G') + full_cds_seq.upper().count('C')
                gc_content = gc_count / len(full_cds_seq) if len(full_cds_seq) > 0 else 0
                
                primary_transcripts[current_transcript_id] = {
                    'gene_id': current_base_gene, 
                    'cds_gc': gc_content,
                    'promoter_gc': None,
                    'promoter_source': None,
                    'in_gff': False 
                }
            
            # Extract identifiers
            current_transcript_id = line[1:].split()[0]
            current_base_gene = current_transcript_id.split('_')[0]
            current_seq = []
        else:
            current_seq.append(line)

# Process the final sequence in the file
if current_transcript_id and current_seq and current_base_gene in required_genes:
    full_cds_seq = "".join(current_seq)
    gc_count = full_cds_seq.upper().count('G') + full_cds_seq.upper().count('C')
    gc_content = gc_count / len(full_cds_seq) if len(full_cds_seq) > 0 else 0
    primary_transcripts[current_transcript_id] = {
        'gene_id': current_base_gene,
        'cds_gc': gc_content,
        'promoter_gc': None,
        'promoter_source': None,
        'in_gff': False
    }

print(f"Loaded {len(primary_transcripts)} required primary transcripts from Phytozome.")

# -----------------------------------------------------------------------------
# BLOCK 2: Prioritize Empirical Jores Dataset for Promoters
# -----------------------------------------------------------------------------
print("Mapping empirical promoter GC content from Jores et al. dataset...")
jores_match_count = 0
gene_to_transcript = {data['gene_id']: t_id for t_id, data in primary_transcripts.items()}

with open(jores_file, "r", encoding='utf-8-sig') as j_in:
    reader = csv.DictReader(j_in)
    for row in reader:
        if row.get('species_sup1') == 'Maize':
            gene_id = row.get('gene_sup1')
            if gene_id in gene_to_transcript:
                t_id = gene_to_transcript[gene_id]
                try:
                    gc_val = float(row.get('GC_sup1'))
                    primary_transcripts[t_id]['promoter_gc'] = gc_val
                    primary_transcripts[t_id]['promoter_source'] = 'Jores'
                    jores_match_count += 1
                except (ValueError, TypeError):
                    continue

print(f"Mapped {jores_match_count} empirical promoters. {len(required_genes) - jores_match_count} require fallback extraction.")

# -----------------------------------------------------------------------------
# BLOCK 3: Load Master Genomic FASTA for Fallback Extraction
# -----------------------------------------------------------------------------
print("Reading Master Genomic FASTA file...")
seq = {}
with open(fasta_file, "r") as inFile1:
    current_chr = ""
    for line in inFile1:
        line = line.strip()
        if line.startswith('>'):
            current_chr = line.strip('>').split()[0]
            seq[current_chr] = []
        else:
            seq[current_chr].append(line)

for chrom in seq:
    seq[chrom] = "".join(seq[chrom])

# -----------------------------------------------------------------------------
# BLOCK 4: Process GFF3 (Flag Presence & Extract Missing Promoters)
# -----------------------------------------------------------------------------
print("Processing GFF3 file for spatial fallback extraction (-165 to +5)...")
extracted_count = 0

with open(gff_file, "r") as inFile2:
    for line in inFile2:
        if line.startswith("#"):
            continue
            
        fields = line.strip().split("\t")
        if len(fields) < 9:
            continue
            
        if fields[2] in ["mRNA", "transcript"]:
            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            
            attr_str = fields[8]
            try:
                id_field = [x for x in attr_str.split(";") if x.startswith("ID=")][0]
                raw_id = id_field.split("=")[1]
                transcript_id = raw_id.replace("transcript:", "").replace("mRNA:", "")
            except IndexError:
                continue
                
            if transcript_id in primary_transcripts:
                primary_transcripts[transcript_id]['in_gff'] = True
                
                # Skip if already solved by Jores
                if primary_transcripts[transcript_id]['promoter_source'] == 'Jores':
                    continue
                
                # Fallback Extraction
                if chrom not in seq:
                    continue
                    
                promoter_seq = ""
                if strand == "+":
                    tss_index = start - 1
                    extract_start = tss_index - 165
                    extract_end = tss_index + 5
                    if extract_start >= 0 and extract_end <= len(seq[chrom]):
                        promoter_seq = seq[chrom][extract_start:extract_end]
                        
                elif strand == "-":
                    tss_index = end - 1
                    extract_start = tss_index - 4
                    extract_end = tss_index + 166
                    if extract_start >= 0 and extract_end <= len(seq[chrom]):
                        promoter_seq = seq[chrom][extract_start:extract_end][::-1].translate(complement_map)
                
                if len(promoter_seq) == 170:
                    primary_transcripts[transcript_id]['promoter_gc'] = (promoter_seq.upper().count('G') + promoter_seq.upper().count('C')) / 170.0
                    primary_transcripts[transcript_id]['promoter_source'] = 'Extracted'
                    extracted_count += 1

print(f"Computationally extracted {extracted_count} missing promoters from reference genome.")

# -----------------------------------------------------------------------------
# BLOCK 5: Deferred Logging of Critical Mismatches
# -----------------------------------------------------------------------------
print("Evaluating final annotation dropout...")
mismatch_count = 0
critical_dropout_count = 0

with open(log_file, "w") as logOut:
    logOut.write("### CRITICAL MISMATCH LOG: Target Genes Missing from GFF3 Annotations ###\n\n")
    
    for t_id, data in primary_transcripts.items():
        if not data['in_gff']:
            mismatch_count += 1
            if data['promoter_source'] is None:
                critical_dropout_count += 1
                logOut.write(f"CRITICAL DROPOUT: ID '{t_id}' (Base: {data['gene_id']}) is missing from BOTH Jores and GFF3. Will be dropped from matrix.\n")
            else:
                logOut.write(f"WARNING: ID '{t_id}' (Base: {data['gene_id']}) missing from GFF3, but SAVED by Jores empirical data.\n")

    logOut.write(f"\n--- Summary ---\n")
    logOut.write(f"Total target genes physically missing from GFF3: {mismatch_count}\n")
    logOut.write(f"Absolute matrix dropout (Missing from GFF3 AND Jores): {critical_dropout_count}\n")

# -----------------------------------------------------------------------------
# BLOCK 6: Write Merged Output Data
# -----------------------------------------------------------------------------
print("Writing finalized merged metrics to CSV...")
written_count = 0

with open(out_file, "w") as outCSV:
    outCSV.write("Gene_ID,Transcript_ID,Promoter_GC,Gene_CDS_GC,Promoter_Source\n")
    
    for t_id, metrics in primary_transcripts.items():
        # Only write rows that successfully secured BOTH features
        if metrics['promoter_gc'] is not None and metrics['cds_gc'] is not None:
            outCSV.write(f"{metrics['gene_id']},{t_id},{metrics['promoter_gc']:.6f},{metrics['cds_gc']:.6f},{metrics['promoter_source']}\n")
            written_count += 1

print(f"Pipeline complete. Successfully secured metrics for {written_count} out of {len(required_genes)} target genes.")
