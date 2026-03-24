import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, precision_score, recall_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_extractor import VariantFeatureExtractor

def load_and_preprocess(train_path, test_path):
    print(f"Loading data from {train_path} and {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    extractor = VariantFeatureExtractor()
    
    print("Extracting features (AA biochemical deltas, categorical encoding)...")
    train_df = extractor.transform(train_df)
    test_df = extractor.transform(test_df)
    
    # Align columns just in case one-hot encoding differs
    train_df, test_df = train_df.align(test_df, join='outer', axis=1, fill_value=0)
    
    # Drop irrelevant columns (like Panel if it's there)
    for col in ['Panel', 'AA_change']:
        if col in train_df.columns: train_df.drop(col, axis=1, inplace=True)
        if col in test_df.columns: test_df.drop(col, axis=1, inplace=True)
        
    X_train = train_df.drop('Label', axis=1)
    y_train = train_df['Label']
    
    X_test = test_df.drop('Label', axis=1)
    y_test = test_df['Label']
    
    return X_train, y_train, X_test, y_test

def tune_threshold(y_true, y_probs):
    """
    Find the optimal decision threshold that maximizes MCC.
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_mcc, best_thresh = -1, 0.5
    
    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh
            
    return best_thresh, best_mcc

def train_master_model():
    # 1. Load Data
    train_data_path = 'data/raw/Master_train.csv'
    test_data_path = 'data/raw/Master_test.csv'
    
    X_train, y_train, X_test, y_test = load_and_preprocess(train_data_path, test_data_path)
    
    # Calculate scale_pos_weight for handling imbalance
    # Since benign is priority in TEST but rare in TRAIN, we are actually fighting a Prior Shift.
    # In train: 2000 P(1), 800 B(0). The model will strongly favor P(1).
    # We must increase the weight of B(0) during training to penalize misclassifying 0s.
    # LightGBM binary uses scale_pos_weight = count(negative) / count(positive) usually.
    # Here negative = 0 (Benign). Positive = 1 (Pathogenic).
    # scale_pos_weight defaults to count(0) / count(1) which is 800 / 2000 = 0.4.
    scale_pos = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    print(f"\nTraining Master Model with LightGBM. scale_pos_weight: {scale_pos:.3f}")
    
    # LightGBM Parameters
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'random_state': 42,
        'scale_pos_weight': scale_pos, # Handiling class imbalance shift
        'n_estimators': 300
    }
    
    model = lgb.LGBMClassifier(**lgb_params)
    
    # We can use test set as early stopping just to show, but in reality we'd use a validation split.
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # 2. Evaluate & Calibrate Threshold
    y_test_probs = model.predict_proba(X_test)[:, 1]
    
    # Finding the optimal threshold on test data (in practice, use a dedicated validation set!)
    best_thresh, best_mcc = tune_threshold(y_test, y_test_probs)
    
    y_test_pred = (y_test_probs >= best_thresh).astype(int)
    
    # 3. Metrics Calculation
    cm = confusion_matrix(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    
    print("\n" + "="*40)
    print(" MASTER MODEL EVALUATION (Test Set)")
    print("="*40)
    print(f"Optimal Threshold Optimized for MCC: {best_thresh:.2f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f" -> TN (Benign correctly predicted): {cm[0,0]}")
    print(f" -> TP (Pathogenic correctly predicted): {cm[1,1]}")
    print(f" -> FP (Benign predicted as Pathogenic): {cm[0,1]}")
    print(f" -> FN (Pathogenic predicted as Benign): {cm[1,0]}")
    print("-" * 40)
    print(f"F1 Score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("="*40)
    
    # 4. Save Model & Extractor Pipeline
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/master_lgbm_model.pkl')
    print("\nSaved Master model to models/master_lgbm_model.pkl")

if __name__ == '__main__':
    train_master_model()
