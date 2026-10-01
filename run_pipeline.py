"""
End-to-End Model Training, Evaluation, and Serialization Pipeline.
Trains Frequency & Severity models on FreMTPL2 and exports artifacts for deployment.
"""

import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_raw_data, prepare_integrated_dataset
from src.feature_engineering import build_preprocessor, get_feature_names
from src.models import FrequencyModelSuite, SeverityModelSuite, ActuarialPricingEngine
from src.evaluation import (
    evaluate_frequency_classifier,
    evaluate_severity_regressor,
    poisson_deviance,
    gini_coefficient,
    compute_lift_table
)


def run_full_pipeline():
    print("=" * 70)
    print("INSURANCE RISK INTELLIGENCE PLATFORM - TRAINING PIPELINE")
    print("=" * 70)
    
    start_total = time.time()
    
    # 1. Load Data
    print("\n[Step 1/6] Ingesting and aggregating FreMTPL2 datasets...")
    df_freq, df_sev = load_raw_data()
    df_full, df_severity = prepare_integrated_dataset(df_freq, df_sev)
    print(f"  * Full policies: {len(df_full):,} | Policies with claims: {len(df_severity):,}")
    print(f"  * Claim Frequency: {df_full['ClaimInd'].mean()*100:.2f}%")
    print(f"  * Mean Claim Severity: EUR {df_severity['ClaimAmount_capped'].mean():.2f}")
    
    # 2. Train-Test Split
    print("\n[Step 2/6] Performing Actuarial Train/Test Partitioning...")
    feature_cols = [
        "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density",
        "VehBrand", "VehGas", "Area", "Region"
    ]
    
    # Stratified split on claim indicator
    train_full, test_full = train_test_split(
        df_full, test_size=0.20, random_state=42, stratify=df_full["ClaimInd"]
    )
    print(f"  * Frequency Train: {len(train_full):,} rows | Test: {len(test_full):,} rows")
    
    # Severity split on positive claim records
    train_sev, test_sev = train_test_split(
        df_severity, test_size=0.20, random_state=42
    )
    print(f"  * Severity Train: {len(train_sev):,} rows | Test: {len(test_sev):,} rows")
    
    # 3. Fit Feature Transformation Pipeline
    print("\n[Step 3/6] Fitting Feature Engineering & Encoding Pipeline...")
    pipeline = build_preprocessor()
    X_train_full = pipeline.fit_transform(train_full[feature_cols])
    X_test_full = pipeline.transform(test_full[feature_cols])
    
    X_train_sev = pipeline.transform(train_sev[feature_cols])
    X_test_sev = pipeline.transform(test_sev[feature_cols])
    
    feature_names = get_feature_names(pipeline, feature_cols)
    print(f"  * Transformed feature dimensions: {X_train_full.shape[1]}")
    
    # 4. Train Frequency Models
    print("\n[Step 4/6] Training Frequency Models (Classification & Poisson)...")
    y_train_freq = train_full["ClaimInd"].values
    y_test_freq = test_full["ClaimInd"].values
    
    freq_suite = FrequencyModelSuite(random_state=42)
    freq_models = freq_suite.train_all(X_train_full, y_train_freq, exposure_train=train_full["Exposure"].values)
    
    freq_results = {}
    for name, model in freq_models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_full)[:, 1]
        else:
            # Poisson GLM output is expected frequency count or rate
            rate_pred = model.predict(X_test_full)
            # Probability of at least 1 claim under Poisson: 1 - exp(-lambda * exposure)
            probs = 1.0 - np.exp(-rate_pred * test_full["Exposure"].values)
            probs = np.clip(probs, 0.0, 1.0)
            
        metrics = evaluate_frequency_classifier(y_test_freq, probs)
        freq_results[name] = metrics
        print(f"    * {name} -> ROC-AUC: {metrics['ROC_AUC']:.4f} | PR-AUC: {metrics['PR_AUC']:.4f} | Brier: {metrics['Brier_Score']:.5f}")
        
    # 5. Train Severity Models
    print("\n[Step 5/6] Training Severity Models (Regression conditioned on Claim > 0)...")
    y_train_sev_raw = train_sev["ClaimAmount_capped"].values
    y_train_sev_log = train_sev["LogClaimAmount"].values
    y_test_sev_raw = test_sev["ClaimAmount_capped"].values
    
    sev_suite = SeverityModelSuite(random_state=42)
    sev_models = sev_suite.train_all(X_train_sev, y_train_sev_raw, y_train_sev_log)
    
    sev_results = {}
    for name, model in sev_models.items():
        if name in ["Ridge_LogRegression", "LightGBM_Severity", "XGBoost_Severity"]:
            pred_log = model.predict(X_test_sev)
            pred_cost = np.exp(pred_log)
        else:
            pred_cost = model.predict(X_test_sev)
            pred_cost = np.clip(pred_cost, 10.0, None)
            
        metrics = evaluate_severity_regressor(y_test_sev_raw, pred_cost)
        sev_results[name] = metrics
        print(f"    * {name} -> MAE: EUR {metrics['MAE']} | RMSE: EUR {metrics['RMSE']} | Gamma Dev: {metrics['Gamma_Deviance']}")
        
    # 6. Compound Pure Premium & Risk Stratification
    print("\n[Step 6/6] Computing Pure Premium and Risk Differentiation Metrics...")
    best_freq_model = freq_models["LightGBM_Classifier"]
    best_sev_model = sev_models["LightGBM_Severity"]
    pricing_engine = ActuarialPricingEngine()
    
    # Portfolio Test Predictions
    test_claim_probs = best_freq_model.predict_proba(X_test_full)[:, 1]
    test_sev_preds = np.exp(best_sev_model.predict(X_test_full))
    test_pure_premium = test_claim_probs * test_sev_preds
    
    # Actual observed loss for test set
    test_actual_loss = test_full["ClaimAmount_total"].values
    
    portfolio_gini = gini_coefficient(test_actual_loss, test_pure_premium)
    print(f"  * Portfolio Normalized Gini Index: {portfolio_gini:.4f}")
    
    df_eval = pd.DataFrame({
        "IDpol": test_full["IDpol"].values,
        "Predicted_Pure_Premium": test_pure_premium,
        "Actual_Loss": test_actual_loss
    })
    lift_table = compute_lift_table(df_eval, n_bins=10)
    print("\n  Risk Decile Lift Table:")
    print(lift_table[["Decile", "Policy_Count", "Avg_Predicted_Loss", "Avg_Actual_Loss", "Loss_Capture_Pct"]].to_string(index=False))
    
    # 7. Serialize Artifacts
    print("\nSerializing artifacts for production deployment...")
    joblib.dump(pipeline, "models/preprocessor.pkl")
    joblib.dump(best_freq_model, "models/frequency_classifier.pkl")
    joblib.dump(best_sev_model, "models/severity_regressor.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")
    
    # Save a representative test sample for interactive SHAP and testing (1,000 rows)
    sample_df = test_full.sample(n=1000, random_state=42)
    sample_df.to_csv("models/portfolio_sample.csv", index=False)
    
    metadata = {
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_records": len(df_full),
        "frequency_claim_rate": float(df_full["ClaimInd"].mean()),
        "mean_claim_severity": float(df_severity["ClaimAmount_capped"].mean()),
        "median_claim_severity": float(df_severity["ClaimAmount_capped"].median()),
        "portfolio_gini": float(portfolio_gini),
        "frequency_metrics": freq_results,
        "severity_metrics": sev_results,
        "commercial_pricing_parameters": {
            "fixed_expense": pricing_engine.fixed_expense,
            "variable_expense_ratio": pricing_engine.variable_expense_ratio,
            "profit_margin": pricing_engine.profit_margin,
            "contingency_margin": pricing_engine.contingency_margin
        }
    }
    
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\n Pipeline completed successfully in {time.time() - start_total:.1f} seconds!")
    print(" Artifacts generated:")
    print("  - models/preprocessor.pkl")
    print("  - models/frequency_classifier.pkl")
    print("  - models/severity_regressor.pkl")
    print("  - models/feature_names.pkl")
    print("  - models/portfolio_sample.csv")
    print("  - models/metadata.json")


if __name__ == "__main__":
    run_full_pipeline()
