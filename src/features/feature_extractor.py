import pandas as pd
import numpy as np
import os
import sys

# To allow relative imports if run as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.amino_acid import AA_PROPERTIES

class VariantFeatureExtractor:
    def __init__(self):
        self.aa_props = AA_PROPERTIES
    
    def parse_aa_change(self, aa_str):
        """
        Parses format 'A>V' and returns reference and alternate amino acids.
        If format is invalid or missing, returns (None, None).
        """
        if pd.isna(aa_str) or '>' not in aa_str:
            return None, None
        
        parts = aa_str.split('>')
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        return None, None

    def extract_aa_features(self, df, aa_col='AA_change'):
        """
        Calculates the delta (change) in biochemical properties between Ref and Alt AA.
        """
        if aa_col not in df.columns:
            return df
        
        # New columns
        df['Delta_Hydrophobicity'] = 0.0
        df['Delta_MW'] = 0.0
        df['Delta_pI'] = 0.0
        df['Delta_Volume'] = 0.0
        df['Polarity_Change'] = 0 # 1 if changed, 0 if same
        
        for idx, row in df.iterrows():
            ref, alt = self.parse_aa_change(row[aa_col])
            
            if ref in self.aa_props and alt in self.aa_props:
                ref_p = self.aa_props[ref]
                alt_p = self.aa_props[alt]
                
                df.at[idx, 'Delta_Hydrophobicity'] = alt_p['hydrophobicity'] - ref_p['hydrophobicity']
                df.at[idx, 'Delta_MW'] = alt_p['molecular_weight'] - ref_p['molecular_weight']
                df.at[idx, 'Delta_pI'] = alt_p['pI'] - ref_p['pI']
                df.at[idx, 'Delta_Volume'] = alt_p['volume'] - ref_p['volume']
                df.at[idx, 'Polarity_Change'] = 1 if ref_p['polarity'] != alt_p['polarity'] else 0
                
        return df
        
    def encode_categorical(self, df, cat_cols=None):
        if cat_cols is None:
            cat_cols = [c for c in df.columns if c.startswith('CAT_')]
        
        # Simple One-Hot Encoding for demonstration
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            
        return df

    def transform(self, df):
        """
        Runs the full feature engineering pipeline on tabular data.
        """
        df = df.copy()
        
        # 1. AA Features
        df = self.extract_aa_features(df, aa_col='AA_change')
        
        # 2. Categorical Encoding (CAT_)
        df = self.encode_categorical(df)
        
        # Drop raw AA string
        if 'AA_change' in df.columns:
            df.drop('AA_change', axis=1, inplace=True)
            
        # Drop Panel column if standardizing. We will keep it for routing but drop from model inputs later
        
        return df

if __name__ == '__main__':
    # Test the extractor
    extractor = VariantFeatureExtractor()
    test_df = pd.DataFrame({'AA_change': ['A>V', 'R>D', 'N>Q', 'invalid', np.nan]})
    res = extractor.transform(test_df)
    print(res)
