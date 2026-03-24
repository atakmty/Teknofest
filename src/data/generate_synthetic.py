import pandas as pd
import numpy as np
import os
import random

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_panel_data(n_pathogenic, n_benign, panel_name, is_train=True):
    total = n_pathogenic + n_benign
    
    # Generate labels
    labels = [1] * n_pathogenic + [0] * n_benign
    
    # Generate continuous features (AL_ Frequency)
    # Pathogenic variants usually have lower frequency (rarer)
    al_freq1 = [np.random.beta(1, 20) if l == 1 else np.random.beta(2, 10) for l in labels]
    al_freq2 = [np.random.beta(1, 50) if l == 1 else np.random.beta(2, 5) for l in labels]
    
    # Evolutionary Conservation (EK_ Scores)
    # Pathogenic are highly conserved (higher score)
    ek_score1 = [np.random.normal(0.8, 0.1) if l == 1 else np.random.normal(0.3, 0.2) for l in labels]
    ek_score2 = [np.random.normal(4.5, 0.5) if l == 1 else np.random.normal(1.5, 0.8) for l in labels]
    
    # Categorical (CAT_ variant type etc)
    var_types = ['missense', 'nonsense', 'frameshift', 'splice_site']
    cat_type = [random.choice(var_types) for _ in range(total)]
    
    # Amino Acid Changes (AA)
    # Format: X>Y where X and Y are standard 1-letter codes
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_changes = []
    for l in labels:
        ref = random.choice(amino_acids)
        alt = random.choice([a for a in amino_acids if a != ref])
        aa_changes.append(f"{ref}>{alt}")
        
    df = pd.DataFrame({
        'AL_freq1': al_freq1,
        'AL_freq2': al_freq2,
        'EK_score1': ek_score1,
        'EK_score2': ek_score2,
        'CAT_type': cat_type,
        'AA_change': aa_changes,
        'Panel': [panel_name] * total,
        'Label': labels
    })
    
    # Clip values to realistic ranges
    df['AL_freq1'] = df['AL_freq1'].clip(0, 1)
    df['AL_freq2'] = df['AL_freq2'].clip(0, 1)
    df['EK_score1'] = df['EK_score1'].clip(0, 1)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def generate_datasets():
    os.makedirs('data/raw', exist_ok=True)
    
    # Scenarios defined by user
    scenarios = {
        'Master':    {'train': (2000, 800), 'test': (500, 3000)},
        'Kanser':    {'train': (250, 100),  'test': (100, 500)},
        'PAH':       {'train': (300, 50),   'test': (100, 250)},
        'CFTR':      {'train': (100, 20),   'test': (20, 100)}
    }
    
    for panel, sizes in scenarios.items():
        # Train
        train_p, train_b = sizes['train']
        df_train = generate_panel_data(train_p, train_b, panel, is_train=True)
        df_train.to_csv(f'data/raw/{panel}_train.csv', index=False)
        
        # Test
        test_p, test_b = sizes['test']
        df_test = generate_panel_data(test_p, test_b, panel, is_train=False)
        df_test.to_csv(f'data/raw/{panel}_test.csv', index=False)
        
        print(f"Generated {panel} -> Train: {train_p}P/{train_b}B = {len(df_train)}, Test: {test_p}P/{test_b}B = {len(df_test)}")

if __name__ == '__main__':
    generate_datasets()
