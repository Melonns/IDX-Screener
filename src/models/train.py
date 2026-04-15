import os
import sys
import json
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

def get_feature_cols(df):
    exclude_cols = [
        'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Target_Momentum', 'tr1', 'tr2', 'tr3', 'True_Range'
    ]
    return [c for c in df.columns if c not in exclude_cols]


def get_feature_importance(model, feature_cols):
    raw_importances = model.get_booster().get_score(importance_type='weight')
    mapped = {}
    for name, score in raw_importances.items():
        if name.startswith('f'):
            mapped[feature_cols[int(name[1:])]] = score
        else:
            mapped[name] = score

    for feature in feature_cols:
        mapped.setdefault(feature, 0)

    return sorted(mapped.items(), key=lambda x: x[1], reverse=True)


def print_evaluation(y_true, y_pred, label):
    print(f"\n[{label}] Evaluasi Model:")
    print(classification_report(y_true, y_pred))
    print(f"Precision (Akurasi Prediksi Naik): {precision_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
    print(f"Accuracy Keseluruhan: {accuracy_score(y_true, y_pred) * 100:.2f}%")


def train_model():
    print("Memulai proses training model Momentum & Profit Predictor...")
    df = load_data()
    feature_cols = get_feature_cols(df)
    
    print(f"Menggunakan {len(feature_cols)} fitur teknikal: {feature_cols}")
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['Target_Momentum']
    X_test = test_df[feature_cols]
    y_test = test_df['Target_Momentum']
    
    print(f"\nDistribusi Target pada Training Set sebelum SMOTE:")
    print(y_train.value_counts())
    
    print("\nMenjalankan SMOTE untuk menyeimbangkan ketimpangan data (Class Imbalance)...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Distribusi Target pada Training Set sesudah SMOTE:")
    print(y_train_resampled.value_counts())
    
    print("\nMelatih Model XGBoost pada semua fitur...")
    base_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    base_model.fit(X_train_resampled, y_train_resampled)
    
    print_evaluation(y_test, base_model.predict(X_test), label='Baseline Full Features')
    
    sorted_importances = get_feature_importance(base_model, feature_cols)
    print("\nRanking fitur penting berdasarkan XGBoost weight:")
    for feature, score in sorted_importances:
        print(f"  {feature}: {score}")
    
    selected_features = feature_cols
    final_model = base_model
    if config.TOP_FEATURE_SELECTION and 0 < config.TOP_FEATURE_COUNT < len(feature_cols):
        selected_features = [f for f, _ in sorted_importances[:config.TOP_FEATURE_COUNT]]
        print(f"\nMenggunakan top {config.TOP_FEATURE_COUNT} fitur teratas: {selected_features}")
        final_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
        final_model.fit(X_train_resampled[selected_features], y_train_resampled)
        print_evaluation(y_test, final_model.predict(X_test[selected_features]), label=f'Top {config.TOP_FEATURE_COUNT} Features')
    
    model_dir = os.path.join(config.BASE_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_filename = f"xgboost_momentum_{timestamp}.joblib"
    model_path = os.path.join(model_dir, model_filename)
    joblib.dump(final_model, model_path)
    
    feature_path = os.path.join(model_dir, f"selected_features_{timestamp}.json")
    with open(feature_path, 'w', encoding='utf-8') as f:
        json.dump(selected_features, f, indent=2)
    
    print(f"\nModel berhasil disimpan ke: {model_path}")
    print(f"Selected features disimpan ke: {feature_path}")

if __name__ == "__main__":
    train_model()
