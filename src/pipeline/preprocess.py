import argparse
import pandas as pd
import sys
import os

# Add parent directory to path to allow importing from features
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_extractor import VariantFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description="Preprocess TEKNOFEST Variant Data")
    parser.add_argument('--input', type=str, required=True, help='Path to raw synthetic CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the processed CSV file')
    args = parser.parse_args()
    
    print(f"Reading input file: {args.input}")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: Could not find file {args.input}")
        sys.exit(1)
        
    print("Applying Variant Feature Extractor...")
    extractor = VariantFeatureExtractor()
    processed_df = extractor.transform(df)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    processed_df.to_csv(args.output, index=False)
    print(f"Successfully processed {len(processed_df)} records.")
    print(f"Saved to: {args.output}")

if __name__ == '__main__':
    main()
