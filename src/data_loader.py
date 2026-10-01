"""
Data Loading and Preprocessing Module for Insurance Risk Intelligence Platform.
Dataset: French Motor Third-Party Liability (freMTPL2)
"""

import pandas as pd
import numpy as np
import os


def load_raw_data(data_dir: str = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw frequency and severity datasets from CSV files.
    """
    freq_path = os.path.join(data_dir, "freMTPL2freq.csv")
    sev_path = os.path.join(data_dir, "freMTPL2sev.csv")
    
    if not os.path.exists(freq_path) or not os.path.exists(sev_path):
        raise FileNotFoundError(f"Missing required CSV files in {data_dir}")
        
    df_freq = pd.read_csv(freq_path)
    df_sev = pd.read_csv(sev_path)
    
    # Ensure IDpol is integer
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_sev["IDpol"] = df_sev["IDpol"].astype(int)
    
    return df_freq, df_sev


def prepare_integrated_dataset(
    df_freq: pd.DataFrame, 
    df_sev: pd.DataFrame,
    cap_exposure: bool = True,
    severity_outlier_quantile: float = 0.995
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans, aggregates, and joins frequency and severity data.
    
    Returns:
    - df_full: Complete dataset with policy characteristics, exposure, claim count, 
               binary claim indicator, and total claim amount (0 if no claim).
    - df_severity: Filtered dataset for policies with positive paid claims (for severity modeling).
    """
    df_f = df_freq.copy()
    df_s = df_sev.copy()
    
    # 1. Exposure handling (standard actuarial practice: cap exposure at 1.0 year)
    if cap_exposure:
        df_f["Exposure"] = df_f["Exposure"].clip(lower=0.001, upper=1.0)
    else:
        df_f["Exposure"] = df_f["Exposure"].clip(lower=0.001)
        
    # 2. Aggregate severity records by IDpol
    # A single policy can experience multiple claims in a single policy year
    sev_agg = df_s.groupby("IDpol").agg(
        ClaimAmount_total=("ClaimAmount", "sum"),
        ClaimAmount_mean=("ClaimAmount", "mean"),
        ClaimCount_sev=("ClaimAmount", "count")
    ).reset_index()
    
    # 3. Left join frequency with aggregated severity
    df_full = pd.merge(df_f, sev_agg, on="IDpol", how="left")
    
    # 4. Fill unobserved claims with 0
    df_full["ClaimAmount_total"] = df_full["ClaimAmount_total"].fillna(0.0)
    df_full["ClaimAmount_mean"] = df_full["ClaimAmount_mean"].fillna(0.0)
    df_full["ClaimCount_sev"] = df_full["ClaimCount_sev"].fillna(0).astype(int)
    
    # 5. Define key modeling targets
    # Binary claim occurrence indicator (0 = No Claim, 1 = Claim Occurred)
    df_full["ClaimInd"] = (df_full["ClaimNb"] > 0).astype(int)
    
    # Claim frequency rate per unit exposure
    df_full["ClaimRate"] = df_full["ClaimNb"] / df_full["Exposure"]
    
    # 6. Extract severity subset (claims with positive payout)
    # Filter policies with recorded positive claim amounts
    df_severity = df_full[df_full["ClaimAmount_total"] > 0].copy()
    
    # Actuarial outlier treatment: Cap extreme bodily injury / catastrophe outliers at quantile threshold
    cap_value = df_severity["ClaimAmount_mean"].quantile(severity_outlier_quantile)
    df_severity["ClaimAmount_capped"] = df_severity["ClaimAmount_mean"].clip(upper=cap_value)
    
    # Log-transformed severity for regression stability
    df_severity["LogClaimAmount"] = np.log(df_severity["ClaimAmount_capped"])
    
    return df_full, df_severity


if __name__ == "__main__":
    freq, sev = load_raw_data()
    full, severity_subset = prepare_integrated_dataset(freq, sev)
    print(f"Full dataset shape: {full.shape}")
    print(f"Claim occurrence rate: {full['ClaimInd'].mean()*100:.2f}%")
    print(f"Severity subset records: {severity_subset.shape[0]}")
    print(f"Mean claim amount (raw): EUR {severity_subset['ClaimAmount_mean'].mean():.2f}")
    print(f"Median claim amount: EUR {severity_subset['ClaimAmount_mean'].median():.2f}")

