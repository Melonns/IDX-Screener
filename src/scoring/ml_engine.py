import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

from ml.explainer import SHAPExplainer

class MLScoringEngine:
    def __init__(self, model_path: str):
        """
        Initialize the ML Inference Engine.
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.explainer = SHAPExplainer(self.model)
        
        # We need the exact features used in training
        self.feature_cols = [
            'rsi_14', 'macd_diff', 'macd', 'macd_signal', 'volume_ratio_20d',
            'ema_9_dist', 'ema_21_dist', 'ema_50_dist', 'bb_pct_b', 'bb_width',
            'rsi_14_cs_rank', 'macd_diff_cs_rank', 'volume_ratio_20d_cs_rank',
            'bb_pct_b_cs_rank', 'bb_width_cs_rank'
        ]
        self.features_to_rank = ['rsi_14', 'macd_diff', 'volume_ratio_20d', 'bb_pct_b', 'bb_width']

    def score_all(self, df_cross_sectional: pd.DataFrame, today: str = None) -> dict:
        """
        Score all tickers on a specific day using Cross-Sectional features.
        
        Args:
            df_cross_sectional: DataFrame containing prices and indicators for multiple tickers on a specific day.
            today: The date string to score. If None, uses the last available date in the DataFrame.
            
        Returns:
            Dictionary mapping ticker to its scoring result (probability and breakdown).
        """
        if df_cross_sectional.empty:
            return {}
            
        if today is None:
            today = df_cross_sectional.index.get_level_values(0).max()
            if isinstance(today, pd.Timestamp):
                today = today.strftime('%Y-%m-%d')
                
        # Filter data for 'today'
        try:
            if isinstance(df_cross_sectional.index, pd.MultiIndex):
                day_data = df_cross_sectional.xs(today, level=0).copy()
            elif isinstance(df_cross_sectional.index, pd.DatetimeIndex):
                # Using .loc with a string date works seamlessly on DatetimeIndex
                day_data = df_cross_sectional.loc[[today]].copy()
            else:
                day_data = df_cross_sectional[df_cross_sectional['date'] == today].copy()
                if 'ticker' in day_data.columns:
                    day_data = day_data.set_index('ticker')
        except KeyError:
            return {}
            
        if day_data.empty:
            return {}
            
        # Ensure ticker is the index if it's currently a column
        if 'ticker' in day_data.columns:
            day_data = day_data.set_index('ticker')

        # Calculate normalized features
        day_data['ema_9_dist'] = day_data['Close'] / day_data['ema_9'] - 1
        day_data['ema_21_dist'] = day_data['Close'] / day_data['ema_21'] - 1
        day_data['ema_50_dist'] = day_data['Close'] / day_data['ema_50'] - 1
        
        bb_range = day_data['bb_upper'] - day_data['bb_lower']
        day_data['bb_pct_b'] = np.where(bb_range == 0, 0, (day_data['Close'] - day_data['bb_lower']) / bb_range)
        
        # Calculate cross-sectional ranks
        for f in self.features_to_rank:
            day_data[f'{f}_cs_rank'] = day_data[f].rank(pct=True)
            
        # Predict for each ticker
        results = {}
        for ticker, row in day_data.iterrows():
            X_row = pd.DataFrame([row])[self.feature_cols]
            
            # If any features are missing (NaN), we can't score accurately, but XGBoost handles NaNs natively.
            proba = self.model.predict_proba(X_row)[0][1]
            skor_total = int(proba * 100)
            
            breakdown = self.explainer.get_shap_breakdown(X_row)
            
            sinyal = "BULLISH" if skor_total >= 60 else ("BEARISH" if skor_total <= 40 else "NEUTRAL")
            
            results[ticker] = {
                "kode": ticker,
                "tanggal": today,
                "skor_total": skor_total,
                "sinyal": sinyal,
                "scoring_version": "ml_v1.0",
                "breakdown": breakdown
            }
            
        return results
