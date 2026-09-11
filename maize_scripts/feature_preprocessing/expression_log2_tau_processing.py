import sys
import pandas as pd
import numpy as np

# Usage Check
if len(sys.argv) != 3:
    print("Usage: python expression_log2_tau_processing.py <raw_fpkm.csv> <output_features.csv>")
    sys.exit(1)

input_csv = sys.argv[1]   # Raw FPKM expression matrix (e.g., Expression_correct.csv)
output_csv = sys.argv[2]  # Output path for computed Log2_Average and Tau_Index features

def process_expression_features(input_csv, output_csv):
    """
    Ingests raw FPKM expression data, normalizes it via log2(x+1), and 
    calculates both Expression Magnitude (Log2_Average) and Expression 
    Breadth (Tau_Index) for each individual gene.
    """
    print("Loading raw expression matrix...")
    df = pd.read_csv(input_csv)
    
    # ---------------------------------------------------------
    # BLOCK 1: Isolate Tissues and Transform
    # ---------------------------------------------------------
    # Columns 0-3 are gene metadata (gene_id, start, end, chr).
    # The last column is a linear-scale average provided in the source file;
    # it is not used here since Log2_Average (below) supersedes it.
    tissue_cols = df.columns[4:-1]
    raw_tissues = df[tissue_cols]
    
    log_tissues = np.log2(raw_tissues + 1)
    
    # ---------------------------------------------------------
    # BLOCK 2: Calculate Expression Magnitude (Level)
    # ---------------------------------------------------------
    df['Log2_Average'] = log_tissues.mean(axis=1)
    
    # ---------------------------------------------------------
    # BLOCK 3: Calculate Expression Breadth (Tau Index)
    # ---------------------------------------------------------
    N = len(tissue_cols) # N = 24 tissues
    max_expr = log_tissues.max(axis=1)
    df['Tau_Index'] = np.nan
    
    # BIOLOGICAL EDGE CASE: Genes with 0 FPKM across ALL 24 tissues.
    # Their max_expr is 0. Attempting to calculate Tau will cause a 
    # division-by-zero error. We create a mask to isolate expressed genes.
    expressed_mask = max_expr > 0
    normalized_expr = log_tissues[expressed_mask].div(max_expr[expressed_mask], axis=0)
    tau_values = (1 - normalized_expr).sum(axis=1) / (N - 1)
    df.loc[expressed_mask, 'Tau_Index'] = tau_values
    
    # ---------------------------------------------------------
    # BLOCK 4: Export Clean Feature Matrix
    # ---------------------------------------------------------
    final_features = df[['gene_id', 'Log2_Average', 'Tau_Index']]
    
    print("Saving processed features...")
    final_features.to_csv(output_csv, index=False)
    
    print(f"Successfully processed {len(df)} genes.")
    print(f"Output saved to: {output_csv}")
    
    return final_features

if __name__ == "__main__":
    process_expression_features(input_csv, output_csv)