import argparse
import pandas as pd
import numpy as np
import sys
import os
import joblib
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Scikit-learn çıktılarını Numpy dizisi (Array) yerine Pandas DataFrame olarak tutmak için (Kolon isimleri kaybolmasın)
sklearn.set_config(transform_output="pandas")

# Add parent directory to path to allow importing from features
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_extractor import VariantFeatureExtractor

class CollinearDropper(BaseEstimator, TransformerMixin):
    """
    Korelasyon analizi: Birbiriyle çok yüksek korelasyona sahip (örneğin %95 üzeri)
    özelliklerden birini eleyerek modelin ezber yapmasını (collinearity) önler.
    Kör Analiz'e yardımcı olarak gereksiz veri şişmesini engeller.
    """
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.drop_cols_ = []
        
    def fit(self, X, y=None):
        # Yalnızca sayısal sütunlarda Mutlak Değer (Abs) Korelasyon hesapla
        corr_matrix = X.corr(numeric_only=True).abs()
        # Sadece üçgen matrisin üst yarısındaki tekrarları al
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.drop_cols_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self
        
    def transform(self, X):
        return X.drop(columns=[c for c in self.drop_cols_ if c in X.columns])

def build_pipeline():
    """
    Uçtan uca eksiksiz makine öğrenmesi hazırlık boru hattı.
    1. VariantFeatureExtractor: AA değişimi, Imputation, One-Hot Encoding
    2. VarianceThreshold: Sıfır varyanslı (hep aynı değer) işe yaramaz sütunları atar
    3. CollinearDropper: Birbiriyle %95 üzeri aynı giden kopyaları atar
    4. RobustScaler: Aykırı değerlere dirençli (Outlier Resistant) Normalizasyon yapar
    """
    return Pipeline([
        ('feature_extraction', VariantFeatureExtractor()),
        ('variance_filter', VarianceThreshold(threshold=0.0)),
        ('collinear_dropper', CollinearDropper(threshold=0.95)),
        ('scaler', RobustScaler())
    ])

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
        print("Training End-to-End ML Pipeline (Fitting data)...")
        pipeline = build_pipeline()
        processed_df = pipeline.fit_transform(df)
        
        # Save state to prevent data leakage in tests
        joblib.dump(pipeline, args.model_path)
        print(f"Saved full Pipeline state to: {args.model_path}")
    else:
        print("Loading Pre-trained Pipeline...")
        try:
            pipeline = joblib.load(args.model_path)
        except Exception as e:
            print(f"Error loading model state {args.model_path}: {e}")
            sys.exit(1)
            
        processed_df = pipeline.transform(df)
    
    # Save results
    processed_df.to_csv(args.output, index=False)
    print(f"Successfully processed {len(processed_df)} records.")
    print(f"Saved data to: {args.output}")

if __name__ == '__main__':
    main()
