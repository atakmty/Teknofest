import gzip
import csv
import re
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# 3-letter to 1-letter amino acid code mapping
AA_MAP = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D',
    'Cys': 'C', 'Gln': 'Q', 'Glu': 'E', 'Gly': 'G',
    'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K',
    'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S',
    'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
}

def extract_aa_change(name_str):
    # Looking for patterns like p.Arg34Gln
    match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', name_str)
    if match:
        ref = match.group(1)
        alt = match.group(3)
        if ref in AA_MAP and alt in AA_MAP:
            return f"{AA_MAP[ref]}>{AA_MAP[alt]}"
    return None

def generate_dataset(input_file, train_output, test_output, max_samples=10000):
    print("Generating synthetic dataset from variant_summary.txt.gz...")
    dataset = []
    
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        # Read header
        header = None
        for line in f:
            if line.startswith('#'):
                header = line.strip('#\n').split('\t')
                break
        
        if not header:
            print("Error: Could not find header in variant_summary.txt.gz")
            return
            
        try:
            type_idx = header.index('Type')
            name_idx = header.index('Name')
            sig_idx = header.index('ClinicalSignificance')
        except ValueError as e:
            print(f"Error finding columns: {e}")
            return

        for line in f:
            if len(dataset) >= max_samples:
                break
                
            cols = line.strip('\n').split('\t')
            if len(cols) <= sig_idx: continue
            
            var_type = cols[type_idx]
            name = cols[name_idx]
            sig = cols[sig_idx].lower()
            
            if var_type != 'single nucleotide variant':
                continue
                
            # Filter for clear pathogenic or benign
            if 'pathogenic' in sig and 'benign' not in sig and 'uncertain' not in sig and 'conflicting' not in sig:
                label = 1 # Pathogenic
            elif 'benign' in sig and 'pathogenic' not in sig and 'uncertain' not in sig and 'conflicting' not in sig:
                label = 0 # Benign
            else:
                continue
                
            aa_change = extract_aa_change(name)
            if not aa_change:
                continue
                
            # Synthesize EK_ and AL_ scores
            # If pathogenic (label=1), EV conservation is high (EK scores high), frequency is rare (AL low)
            # If benign (label=0), EV conservation is low, frequency is common
            if label == 1:
                ek_revel = min(1.0, max(0.0, random.gauss(0.8, 0.15)))
                ek_sift = min(1.0, max(0.0, random.gauss(0.9, 0.1)))
                ek_polyphen = min(1.0, max(0.0, random.gauss(0.85, 0.15)))
                al_gnomad = max(0.0, random.gauss(0.0001, 0.0005))
                al_1000g = max(0.0, random.gauss(0.0002, 0.0005))
            else:
                ek_revel = min(1.0, max(0.0, random.gauss(0.2, 0.2)))
                ek_sift = min(1.0, max(0.0, random.gauss(0.1, 0.15)))
                ek_polyphen = min(1.0, max(0.0, random.gauss(0.15, 0.2)))
                al_gnomad = min(1.0, max(0.0, random.gauss(0.05, 0.02)))
                al_1000g = min(1.0, max(0.0, random.gauss(0.06, 0.03)))
                
            dataset.append({
                'AA_change': aa_change,
                'CAT_type': 'missense',
                'EK_REVEL': ek_revel,
                'EK_SIFT': ek_sift,
                'EK_Polyphen': ek_polyphen,
                'AL_gnomAD_AF': al_gnomad,
                'AL_1000G_AF': al_1000g,
                'Label': label
            })
            
    if not dataset:
        print("Error: No variants extracted.")
        return
        
    df = pd.DataFrame(dataset)
    print(f"Extracted {len(df)} variants.")
    print(f"Pathogenic: {sum(df['Label'] == 1)}, Benign: {sum(df['Label'] == 0)}")
    
    # Stratified split 80% train, 20% test
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)
    
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    print(f"Saved {len(train_df)} rows to {train_output}")
    print(f"Saved {len(test_df)} rows to {test_output}")

if __name__ == "__main__":
    generate_dataset(
        input_file="variant_summary.txt.gz",
        train_output="data/raw/Master_train.csv",
        test_output="data/raw/Master_test.csv"
    )
