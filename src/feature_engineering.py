"""
Feature Engineering and Pipeline Transformation Module.
Standardizes tabular features for insurance risk modeling (Frequency & Severity).
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class InsuranceFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for actuarial feature derivations:
    - Log transforms for skewed variables (Density)
    - Age binning and interaction terms
    - Risk index flags
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # 1. Log density (linearize exponential population densities)
        df["LogDensity"] = np.log1p(df["Density"].clip(lower=0))
        
        # 2. Risk interaction terms
        df["YoungDriver"] = (df["DrivAge"] < 25).astype(int)
        df["SeniorDriver"] = (df["DrivAge"] >= 70).astype(int)
        df["HighBonusMalus"] = (df["BonusMalus"] > 100).astype(int)
        df["MaxDiscountBonus"] = (df["BonusMalus"] <= 50).astype(int)
        
        # Dangerous driver-vehicle interaction (Young driver with high vehicle power)
        df["YoungHighPower"] = ((df["DrivAge"] < 25) & (df["VehPower"] >= 8)).astype(int)
        
        # Power per driver age ratio proxy
        df["PowerAgeRatio"] = df["VehPower"] / (df["DrivAge"].clip(lower=18))
        
        # Vehicle Age squared (aging curve)
        df["VehAge_sq"] = df["VehAge"] ** 2
        
        return df


def build_preprocessor(categorical_strategy: str = "onehot") -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for all insurance features.
    Handles numeric scaling, categorical encoding, and custom feature engineering.
    """
    numeric_features = [
        "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density",
        "LogDensity", "YoungDriver", "SeniorDriver", "HighBonusMalus",
        "MaxDiscountBonus", "YoungHighPower", "PowerAgeRatio", "VehAge_sq"
    ]
    
    categorical_features = ["VehBrand", "VehGas", "Area", "Region"]
    
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("cat", cat_transformer, categorical_features)
        ],
        remainder="drop"
    )
    
    return Pipeline(steps=[
        ("feature_engineer", InsuranceFeatureEngineer()),
        ("transformer", preprocessor)
    ])


def get_feature_names(fitted_pipeline, original_feature_cols) -> list[str]:
    """
    Extract transformed feature names from the fitted pipeline.
    """
    try:
        transformer = fitted_pipeline.named_steps["transformer"]
        return list(transformer.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out is not available
        return [f"feature_{i}" for i in range(100)]


if __name__ == "__main__":
    try:
        from src.data_loader import load_raw_data, prepare_integrated_dataset
    except ImportError:
        from data_loader import load_raw_data, prepare_integrated_dataset
    freq, sev = load_raw_data()
    full, _ = prepare_integrated_dataset(freq, sev)
    
    feature_cols = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density", "VehBrand", "VehGas", "Area", "Region"]
    pipe = build_preprocessor()
    X_sample = full[feature_cols].head(1000)
    X_trans = pipe.fit_transform(X_sample)
    
    names = get_feature_names(pipe, feature_cols)
    print(f"Sample transformed shape: {X_trans.shape}")
    print(f"Total features created: {len(names)}")
    print(f"First 10 feature names: {names[:10]}")
