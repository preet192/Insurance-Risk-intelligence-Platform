"""
Actuarial & Machine Learning Evaluation Metrics Module.
Calculates performance indicators for Frequency, Severity, and Compound Pure Premium.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, mean_absolute_error, mean_squared_error, r2_score
)


def evaluate_frequency_classifier(y_true, y_prob, threshold: float = 0.5) -> dict:
    """
    Computes classification metrics for claim occurrence P(Claim > 0).
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    
    # Observed vs Expected total claims
    obs_claims = int(np.sum(y_true))
    exp_claims = float(np.sum(y_prob))
    
    return {
        "ROC_AUC": round(float(auc_roc), 4),
        "PR_AUC": round(float(auc_pr), 4),
        "Brier_Score": round(float(brier), 5),
        "Log_Loss": round(float(ll), 4),
        "Observed_Claims": obs_claims,
        "Expected_Claims": round(exp_claims, 1),
        "AE_Ratio": round(exp_claims / max(obs_claims, 1), 4) # Actual vs Expected ratio
    }


def poisson_deviance(y_true, y_pred, exposure=None) -> float:
    """
    Computes mean unit Poisson deviance:
    2 * sum( y * log(y / y_hat) - (y - y_hat) ) / n
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 1e-9, None)
    
    if exposure is not None:
        y_pred = y_pred * np.asarray(exposure, dtype=float)
        
    dev = np.zeros_like(y_true)
    pos = y_true > 0
    dev[pos] = y_true[pos] * np.log(y_true[pos] / y_pred[pos]) - (y_true[pos] - y_pred[pos])
    dev[~pos] = y_pred[~pos]
    return float(2 * np.mean(dev))


def gamma_deviance(y_true, y_pred) -> float:
    """
    Computes mean Gamma deviance:
    2 * sum( -log(y / y_hat) + (y - y_hat) / y_hat ) / n
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.clip(y_true, 1e-6, None)
    y_pred = np.clip(y_pred, 1e-6, None)
    dev = 2 * (-np.log(y_true / y_pred) + (y_true - y_pred) / y_pred)
    return float(np.mean(dev))


def evaluate_severity_regressor(y_true, y_pred) -> dict:
    """
    Computes regression metrics for claim cost (severity).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    g_dev = gamma_deviance(y_true, y_pred)
    
    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2_Score": round(float(r2), 4),
        "Gamma_Deviance": round(float(g_dev), 4),
        "Mean_Observed": round(float(np.mean(y_true)), 2),
        "Mean_Predicted": round(float(np.mean(y_pred)), 2)
    }


def gini_coefficient(y_true, y_pred) -> float:
    """
    Normalized Gini coefficient (Lorenz curve concentration index).
    Widely used by actuaries to measure risk differentiation power.
    """
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df = df.sort_values("pred", ascending=False).reset_index(drop=True)
    
    cum_true = np.cumsum(df["true"]) / np.sum(df["true"])
    cum_pop = np.arange(1, len(df) + 1) / len(df)
    
    # Area under the model Lorenz curve
    lorenz_model = np.sum(cum_true) / len(df)
    
    # Area under the perfect ranking curve
    df_perf = df.sort_values("true", ascending=False).reset_index(drop=True)
    cum_perf = np.cumsum(df_perf["true"]) / np.sum(df_perf["true"])
    lorenz_perf = np.sum(cum_perf) / len(df)
    
    # Area under diagonal (random baseline = 0.5)
    gini = (lorenz_model - 0.5) / (lorenz_perf - 0.5)
    return float(gini)


def compute_lift_table(df_eval: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Segments policyholders into deciles of predicted risk to verify monotonicity of risk separation.
    """
    df = df_eval.copy()
    df["Decile"] = pd.qcut(df["Predicted_Pure_Premium"], q=n_bins, labels=[f"D{i+1}" for i in range(n_bins)])
    
    lift = df.groupby("Decile", observed=False).agg(
        Policy_Count=("IDpol", "count"),
        Avg_Predicted_Loss=("Predicted_Pure_Premium", "mean"),
        Avg_Actual_Loss=("Actual_Loss", "mean"),
        Total_Observed_Loss=("Actual_Loss", "sum")
    ).reset_index()
    
    lift["Loss_Capture_Pct"] = (lift["Total_Observed_Loss"] / lift["Total_Observed_Loss"].sum()) * 100
    return lift

