import shap
import numpy as np
import pandas as pd

class SHAPExplainer:
    def __init__(self, model):
        """
        Initialize the explainer with a trained XGBoost model.
        """
        self.model = model
        # For tree models, TreeExplainer is exactly what we need
        self.explainer = shap.TreeExplainer(self.model)

    def get_shap_breakdown(self, X_row: pd.DataFrame) -> list:
        """
        Generate a breakdown of how each feature contributed to the final probability.
        
        Args:
            X_row: A single row DataFrame containing the features.
            
        Returns:
            list of dicts containing the breakdown, similar to Phase 1 rule-based engine.
        """
        # SHAP values in log-odds margin
        shap_values = self.explainer.shap_values(X_row)
        base_value = self.explainer.expected_value
        
        # We know for binary classification in xgboost, expected_value is an array of size 1 or float
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
            
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # positive class for some estimators
            
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0] # Take the first (and only) row
            
        # Convert log-odds contribution to an approximate percentage contribution 
        # (This is an approximation for explainability, as sigmoid is non-linear)
        # We can map the raw log-odds impact into a readable format.
        
        breakdown = []
        features = X_row.columns
        values = X_row.iloc[0].values
        
        for feature, val, shap_val in zip(features, values, shap_values):
            # Only include features that had a meaningful impact (abs shap > 0.01)
            if abs(shap_val) > 0.01:
                # Convert numeric value to a readable string
                if "rank" in feature:
                    val_str = f"Top {int((1-val)*100)}%" if val > 0.5 else f"Bottom {int(val*100)}%"
                else:
                    val_str = f"{val:.2f}"
                
                # Determine narrative
                impact_dir = "Meningkatkan" if shap_val > 0 else "Menurunkan"
                
                breakdown.append({
                    "indikator": feature,
                    "nilai": val_str,
                    "kontribusi": f"{impact_dir} probabilitas (SHAP: {shap_val:+.2f})",
                    "skor": round(float(shap_val), 3), 
                    "maks": 0 # Not applicable for ML
                })
                
        # Sort by absolute impact (highest impact first)
        breakdown = sorted(breakdown, key=lambda x: abs(x["skor"]), reverse=True)
        return breakdown
