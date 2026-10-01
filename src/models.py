"""
Model Architecture and Training Module for Claim Frequency & Severity.
Implements Two-Part Frequency-Severity Modeling, Pure Premium, and Commercial Pricing Engine.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, PoissonRegressor, GammaRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor


class FrequencyModelSuite:
    """
    Suites for Claim Frequency:
    - Logistic Regression (Baseline)
    - Poisson GLM (Baseline Actuarial GLM)
    - LightGBM Classifier (Primary Production Model)
    - XGBoost Classifier (Benchmark)
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            "Logistic_Regression": LogisticRegression(
                max_iter=1000, 
                class_weight="balanced", 
                random_state=random_state
            ),
            "Poisson_GLM": PoissonRegressor(
                max_iter=500, 
                alpha=1e-4
            ),
            "LightGBM_Classifier": LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1
            ),
            "XGBoost_Classifier": XGBClassifier(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="logloss"
            )
        }

    def train_all(self, X_train, y_train, exposure_train=None):
        fitted_models = {}
        for name, model in self.models.items():
            print(f"  Training {name}...")
            if name == "Poisson_GLM":
                # Poisson GLM takes sample_weight=exposure and y = ClaimNb
                # or rate = ClaimNb / exposure
                if exposure_train is not None:
                    model.fit(X_train, y_train, sample_weight=exposure_train)
                else:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            fitted_models[name] = model
        return fitted_models


class SeverityModelSuite:
    """
    Suites for Claim Severity (conditioned on Claim > 0):
    - Ridge Regression on Log-Loss (Baseline)
    - Gamma GLM (Actuarial Standard)
    - LightGBM Regressor (Primary Production Model)
    - XGBoost Regressor (Benchmark)
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            "Ridge_LogRegression": Ridge(alpha=1.0, random_state=random_state),
            "Gamma_GLM": GammaRegressor(max_iter=500, alpha=1e-3),
            "LightGBM_Severity": LGBMRegressor(
                n_estimators=120,
                learning_rate=0.03,
                num_leaves=25,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="regression",
                random_state=random_state,
                n_jobs=-1,
                verbose=-1
            ),
            "XGBoost_Severity": XGBRegressor(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1
            )
        }

    def train_all(self, X_train_sev, y_train_sev, y_train_log_sev):
        fitted_models = {}
        for name, model in self.models.items():
            print(f"  Training {name}...")
            if name == "Ridge_LogRegression":
                model.fit(X_train_sev, y_train_log_sev)
            elif name == "Gamma_GLM":
                model.fit(X_train_sev, y_train_sev)
            elif name == "LightGBM_Severity":
                model.fit(X_train_sev, y_train_log_sev)
            elif name == "XGBoost_Severity":
                model.fit(X_train_sev, y_train_log_sev)
            fitted_models[name] = model
        return fitted_models


class ActuarialPricingEngine:
    """
    Transforms frequency and severity predictions into Pure Premium and Commercial Premium.
    """
    def __init__(
        self,
        fixed_expense: float = 35.0,        # €35 fixed policy issuance & servicing fee
        variable_expense_ratio: float = 0.18, # 18% commission and operational ratio
        profit_margin: float = 0.05,        # 5% target underwriting profit
        contingency_margin: float = 0.05    # 5% capital buffer for black swan / tail risk
    ):
        self.fixed_expense = fixed_expense
        self.variable_expense_ratio = variable_expense_ratio
        self.profit_margin = profit_margin
        self.contingency_margin = contingency_margin

    def compute_pure_premium(self, claim_probability: float, expected_severity: float) -> float:
        """
        Pure Premium = E[N] * E[Y]
        Expected pure cost of claims per policy.
        """
        return float(claim_probability * expected_severity)

    def compute_commercial_premium(self, pure_premium: float) -> float:
        """
        Calculates recommended commercial gross premium incorporating overhead, acquisition,
        profit loading, and solvency capital buffer.
        """
        denominator = 1.0 - (self.variable_expense_ratio + self.profit_margin)
        gross_premium = ((pure_premium + self.fixed_expense) / denominator) * (1.0 + self.contingency_margin)
        return float(max(gross_premium, 50.0))  # Minimum underwriting floor of €50

    def assign_risk_tier(self, claim_prob: float) -> tuple[str, str]:
        """
        Categorizes risk into actionable underwriting tiers.
        """
        if claim_prob < 0.035:
            return "Tier 1 - Preferred / Low Risk", "green"
        elif claim_prob < 0.075:
            return "Tier 2 - Standard / Moderate Risk", "blue"
        elif claim_prob < 0.14:
            return "Tier 3 - Substandard / High Risk", "orange"
        else:
            return "Tier 4 - Critical / Extreme Risk", "red"

