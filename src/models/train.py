import os
import sys
import pandas as pd
import joblib
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score
from imblearn.over_sampling import SMOTE

# Add src to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_data():
    """Load the processed master dataset."""
    filepath = os.path.join(config.PROCESSED_DATA_DIR, "master_dataset.parquet")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Run build_dataset.py first.")
    
    df = pd.read_parquet(filepath)
    
    # Sort chronologically for time-series integrity
    df = df.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    return df

def train_model():
    print("Memulai proses training model Momentum & Profit Predictor...")
    df = load_data()
    
    # Feature Selection: Exclude non-predictive or target columns
    # We want the model to see the indicators, not the date, ticker, or the future target
    exclude_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target_Momentum', 'tr1', 'tr2', 'tr3', 'True_Range']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Menggunakan {len(feature_cols)} fitur teknikal: {feature_cols}")
    
    # Split data chronologically (e.g. first 80% of time for training, last 20% for testing)
    # This prevents look-ahead bias (data leakage)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['Target_Momentum']
    
    X_test = test_df[feature_cols]
    y_test = test_df['Target_Momentum']
    
    print(f"\nDistribusi Target pada Training Set sebelum SMOTE:")
    print(y_train.value_counts())
    
    # Apply SMOTE to handle class imbalance (Rare Momentum cases)
    print("\nMenjalankan SMOTE untuk menyeimbangkan ketimpangan data (Class Imbalance)...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Distribusi Target pada Training Set sesudah SMOTE:")
    print(y_train_resampled.value_counts())
    
    # Initialize XGBoost target
    print("\nMelatih Model XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    model.fit(X_train_resampled, y_train_resampled)
    
    print("\nEvaluasi Performa Model pada Test Set (Data Masa Depan Tidak Terlihat):")
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    
    # Check predicting capability for "momentum" class (Target = 1)
    prec = precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Precision (Akurasi Prediksi Naik): {prec*100:.2f}%")
    print(f"Accuracy Keseluruhan: {acc*100:.2f}%")
    
    # Save the model
    model_dir = os.path.join(config.BASE_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_filename = f"xgboost_momentum_{timestamp}.joblib"
    model_path = os.path.join(model_dir, model_filename)
    
    joblib.dump(model, model_path)
    print(f"\nModel berhasil disimpan ke: {model_path}")

if __name__ == "__main__":
    train_model()
