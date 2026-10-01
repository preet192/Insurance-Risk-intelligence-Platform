# Insurance Risk Intelligence Platform 🛡️
### Dual-Model Framework for Motor Insurance Claim Frequency, Severity, and Commercial Pricing
**Academic Program:** NMIMS MSc Data Science Capstone Project  
**Domain:** Actuarial Science, Machine Learning, Solvency II / IFRS 17 Insurance Risk Intelligence  
**Dataset:** French Motor Third-Party Liability (`freMTPL2freq` & `freMTPL2sev`)

---

## 🌟 Executive Overview
In property and casualty (P&C) motor insurance, accurately estimating risk is the difference between an underwriting surplus and insurer insolvency. Traditional actuarial practices rely on Generalized Linear Models (GLMs). This project builds a production-grade **Insurance Risk Intelligence Platform** that unifies classical actuarial GLMs with modern **Gradient Boosted Decision Trees (LightGBM & XGBoost)**, **Game-Theoretic Model Explainability (Tree-SHAP)**, and an **Interactive Underwriting Deployment Application**.

The platform tackles two related problems:
1. **Problem A: Claim Frequency ($N$)** - Binary occurrence classification and Poisson rate estimation: *Will a policyholder generate a claim, and at what annual rate?*
2. **Problem B: Claim Severity ($Y$)** - Continuous loss cost regression (conditioned on $N > 0$): *If a claim occurs, how severe will the payout be?*
3. **Compound Pure Premium ($\mathbb{E}[S]$)** - Compound collective risk model: $\mathbb{E}[S] = \mathbb{E}[N] \times \mathbb{E}[Y]$.
4. **Commercial Pricing Engine** - Translating pure loss costs into a market-competitive, solvency-compliant commercial tariff.
5. **Transparent Explainability** - Providing real-time SHAP waterfall charts to justify policyholder pricing under Solvency II fairness mandates.

---

## 📐 Actuarial & Mathematical Architecture

### 1. Compound Collective Risk Model
Total aggregate claim loss $S$ over exposure period $e \in (0, 1]$:
$$S = \sum_{i=1}^{N} Y_i$$
Under the assumption of conditional independence between claim count $N$ and claim severity $Y$:
$$\mathbb{E}[S] = \mathbb{E}[N] \cdot \mathbb{E}[Y]$$

### 2. Frequency Formulation (Problem A)
- **Binary Cross-Entropy (Occurrence)**:
  $$P(N > 0 \mid \mathbf{x}) = \frac{1}{1 + \exp(-\mathbf{x}^\top \boldsymbol{\beta})}$$
- **Poisson Exposure Model**:
  $$N_i \sim \text{Poisson}(\lambda_i \cdot e_i), \quad \log(\lambda_i) = \mathbf{x}_i^\top \boldsymbol{\beta}$$
  Poisson Deviance:
  $$D_{\text{Poisson}}(y, \hat{y}) = 2 \sum \left[ y \log\left(\frac{y}{\hat{y}}\right) - (y - \hat{y}) \right]$$

### 3. Severity Formulation (Problem B)
Conditioned on positive loss payout ($N > 0$):
$$Y_i \sim \text{Gamma}(\alpha, \mu_i), \quad \log(\mu_i) = \mathbf{x}_i^\top \boldsymbol{\gamma}$$
Gamma Deviance:
$$D_{\text{Gamma}}(y, \hat{y}) = 2 \sum \left[ -\log\left(\frac{y}{\hat{y}}\right) + \frac{y - \hat{y}}{\hat{y}} \right]$$

### 4. Commercial Tariff Formula
$$\text{Commercial Premium} = \left( \frac{\text{Pure Premium} + F}{1 - (v + \pi)} \right) \times (1 + c)$$
- $F = \text{EUR } 35$ (Fixed policy issuance and administrative fee)
- $v = 18\%$ (Variable acquisition and claims handling expense ratio)
- $\pi = 5\%$ (Target underwriting profit margin)
- $c = 5\%$ (Solvency contingency buffer for tail catastrophe risk)

---

## 📊 Benchmark Results

### 1. Claim Frequency Performance (Test Set: 135,603 Policies)
| Model Architecture | ROC-AUC | PR-AUC | Brier Score | Log Loss |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Weighted)** | 0.5989 | 0.0767 | 0.2415 | 0.6720 |
| **Poisson GLM (Actuarial Baseline)** | 0.6350 | 0.0987 | 0.0473 | 0.2012 |
| **XGBoost Classifier** | 0.6597 | 0.1156 | 0.0465 | 0.1978 |
| **LightGBM Classifier (Best)** | **0.6631** | **0.1230** | **0.0463** | **0.1969** |

### 2. Claim Severity Performance (Test Set: 4,989 Claims)
| Model Architecture | MAE (€) | RMSE (€) | Gamma Deviance | Mean Predicted (€) |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Log-Regression** | €1,006.91 | €3,093.06 | 1.4654 | €1,281.40 |
| **Gamma GLM** | €1,233.84 | €3,021.66 | **1.1018** | €1,692.10 |
| **XGBoost Regressor** | €995.33 | €3,094.27 | 1.4650 | €1,274.60 |
| **LightGBM Regressor (Best)** | **€996.59** | €3,093.07 | 1.4627 | €1,288.90 |

### 3. Risk Separation & Pure Premium Discrimination
- **Portfolio Normalized Gini Index**: **`0.2522`** (Well within the upper benchmark range of 0.23–0.28 for French MTPL portfolios).
- **Decile Risk Lift**:
  - **Decile 1 (Lowest Risk)**: Avg Predicted Loss = **€18.71**, Actual Loss = **€22.22** (captures only 2.7% of portfolio loss).
  - **Decile 10 (Highest Risk)**: Avg Predicted Loss = **€138.22**, Actual Loss = **€181.94** (captures **22.3% of all portfolio losses**).
  - Demonstrates over **8.2x monotonic risk separation** across portfolio segments!

---

## 🚀 Interactive Streamlit Application

The platform includes an interactive, production-ready web application:

```bash
streamlit run app.py
```

### Key Modules:
1. **🎯 Single Quote Underwriting Engine**:
   - Interactive sliders for Driver Age, Car Age, Bonus-Malus, Vehicle Power, Brand, Fuel, Urban Area, Region, and Exposure.
   - Live computation of Claim Probability ($P$), Severity ($\hat{S}$), Pure Premium ($\text{PP}$), and Commercial Quote.
   - Dynamic Risk Tier badge:
     - 🟢 **Tier 1 (Preferred / Discount Zone)**: $P < 3.5\%$
     - 🔵 **Tier 2 (Standard Market)**: $3.5\% \le P < 7.5\%$
     - 🟡 **Tier 3 (Substandard / Surcharge)**: $7.5\% \le P < 14.0\%$
     - 🔴 **Tier 4 (Critical / High-Risk Decline)**: $P \ge 14.0\%$
   - **Real-time SHAP Waterfall Plot**: Explains the top drivers (e.g. Novice driver penalty, rural discount) for that specific applicant.
2. **📊 Portfolio Risk Analytics**:
   - Portfolio KPI cards, distribution histograms, regional loss heatmaps, and model comparison matrices.
3. **📁 Batch Underwriting Engine**:
   - Upload applicant CSV file or score the pre-loaded 1,000 policy cohort with 1-click CSV export of risk-rated policies.
4. **📖 Actuarial Methodology & Formulas**:
   - Mathematical documentation of compound loss distributions, GLMs, deviances, and Solvency II regulatory compliance.

---

## 📂 Repository Structure

```
├── freMTPL2freq.csv                      # Source Frequency Dataset (678,013 policies)
├── freMTPL2sev.csv                       # Source Severity Dataset (26,639 claims)
├── Insurance_Risk_Claims_Prediction.ipynb # Comprehensive Jupyter Notebook
├── app.py                                # Streamlit Web Application
├── run_pipeline.py                       # End-to-end model training script
├── requirements.txt                      # Dependencies
├── README.md                             # Documentation
├── src/                                  # Modular source code
│   ├── __init__.py
│   ├── data_loader.py                    # Ingestion, cleaning, exposure capping
│   ├── feature_engineering.py            # Log density, interaction terms, pipeline
│   ├── models.py                         # Frequency & Severity models, Pricing Engine
│   ├── evaluation.py                     # Actuarial metrics, Gini, deviances, lift
│   └── explainability.py                 # SHAP TreeExplainer wrapper
└── models/                               # Serialized production artifacts
    ├── preprocessor.pkl                  # Fitted Scikit-Learn transformer
    ├── frequency_classifier.pkl          # Trained LightGBM Classifier
    ├── severity_regressor.pkl            # Trained LightGBM Regressor
    ├── feature_names.pkl                 # Transformed feature dictionary
    ├── portfolio_sample.csv              # Evaluation cohort for batch testing
    └── metadata.json                     # Training metadata and benchmark metrics
```

---

## 🛠️ Quickstart Installation & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Retrain Models & Generate Production Artifacts
```bash
python run_pipeline.py
```

### 3. Launch the Interactive Underwriting Platform
```bash
streamlit run app.py
```

### 4. Open the Jupyter Notebook
Open `Insurance_Risk_Claims_Prediction.ipynb` in VS Code or JupyterLab to inspect the step-by-step actuarial workflows, LaTeX proofs, and visualization charts.

