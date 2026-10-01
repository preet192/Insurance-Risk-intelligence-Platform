"""
Model Explainability Module using SHAP (SHapley Additive exPlanations).
Provides global actuarial risk insights and local policy-level risk attribution.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


class InsuranceExplainer:
    """
    Computes and formats SHAP values for tree-based insurance models.
    """
    def __init__(self, model, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain_instance(self, X_single: np.ndarray, top_n: int = 6) -> list[dict]:
        """
        Explain a single policy quote.
        Returns a sorted list of feature contributions to the prediction.
        """
        if len(X_single.shape) == 1:
            X_single = X_single.reshape(1, -1)
            
        shap_values = self.explainer.shap_values(X_single)
        
        # Handle binary classifier vs regressor output structures
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Positive class SHAP for binary classification
            sv = shap_values[1][0]
            base_val = self.explainer.expected_value[1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
            base_val = self.explainer.expected_value[1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
            sv = shap_values[0]
            base_val = self.explainer.expected_value
        else:
            sv = np.array(shap_values).flatten()
            base_val = self.explainer.expected_value

        contributions = []
        for name, val, shap_val in zip(self.feature_names, X_single[0], sv):
            contributions.append({
                "feature": name.replace("num__", "").replace("cat__", ""),
                "feature_value": float(val),
                "shap_impact": float(shap_val),
                "direction": "Increases Risk" if shap_val > 0 else "Decreases Risk"
            })
            
        # Sort by absolute SHAP impact
        contributions = sorted(contributions, key=lambda x: abs(x["shap_impact"]), reverse=True)
        return contributions[:top_n]

    def compute_shap_matrix(self, X_sample: np.ndarray):
        """
        Computes SHAP values across a background/evaluation cohort.
        """
        shap_values = self.explainer.shap_values(X_sample)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            return shap_values[1]
        return shap_values

