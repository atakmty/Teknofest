import argparse
import pandas as pd
import sys
import os
import joblib

# Add parent directory to path to allow importing from features
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_extractor import VariantFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description="Stateful Preprocess TEKNOFEST Variant Data")
    parser.add_argument('--input', type=str, required=True, help='Path to raw synthetic CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the processed CSV file')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode: fit+transform (train) or transform only (test)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save/load the Extractor .pkl state file')
    args = parser.parse_args()
    
    print(f"Reading input file ({args.mode.upper()} mode): {args.input}")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: Could not find file {args.input}")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    if args.mode == 'train':
        print("Training Variant Feature Extractor (Fitting pipeline)...")
        extractor = VariantFeatureExtractor()
        processed_df = extractor.fit_transform(df)
        
        # Save state to prevent data leakage in tests
        joblib.dump(extractor, args.model_path)
        print(f"Saved Extractor state to: {args.model_path}")
    else:
        print("Loading Pre-trained Variant Feature Extractor...")
        try:
            extractor = joblib.load(args.model_path)
        except Exception as e:
            print(f"Error loading model state {args.model_path}: {e}")
            sys.exit(1)
            
        processed_df = extractor.transform(df)
    
    # Save results
    processed_df.to_csv(args.output, index=False)
    print(f"Successfully processed {len(processed_df)} records.")
    print(f"Saved data to: {args.output}")

if __name__ == '__main__':
    main()
