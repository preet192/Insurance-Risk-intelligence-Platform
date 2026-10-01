"""
Insurance Risk Intelligence Platform - Web Application
Interactive Underwriting, Claim Frequency & Severity Estimation, SHAP Explainability, and Portfolio Analytics.
Built with Streamlit.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import ActuarialPricingEngine
from src.explainability import InsuranceExplainer

# Page Configuration
st.set_page_config(
    page_title="Insurance Risk Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-green {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-blue {
        background-color: #E1EFFE;
        color: #1E429F;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-orange {
        background-color: #FDF6B2;
        color: #723B13;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-red {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_production_artifacts():
    """
    Loads pre-trained pipelines, models, feature names, and metadata.
    """
    preprocessor = joblib.load("models/preprocessor.pkl")
    freq_model = joblib.load("models/frequency_classifier.pkl")
    sev_model = joblib.load("models/severity_regressor.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    
    with open("models/metadata.json", "r") as f:
        metadata = json.load(f)
        
    portfolio_sample = pd.read_csv("models/portfolio_sample.csv")
    pricing_engine = ActuarialPricingEngine()
    explainer = InsuranceExplainer(freq_model, feature_names)
    
    return preprocessor, freq_model, sev_model, feature_names, metadata, portfolio_sample, pricing_engine, explainer


try:
    preprocessor, freq_model, sev_model, feature_names, metadata, portfolio_sample, pricing_engine, explainer = load_production_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure run_pipeline.py has been executed.")
    st.stop()


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/shield.png", width=64)
st.sidebar.title("Risk Navigation")
nav_mode = st.sidebar.radio(
    "Select Workflow Module:",
    [
        "🎯 Underwriting & Quote Engine",
        "📊 Portfolio Risk Analytics",
        "📁 Batch Applicant Scoring",
        "📖 Actuarial Methodology & Formulas"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Pricing Loadings")
fixed_fee = st.sidebar.number_input("Policy Fixed Fee (€)", value=35.0, step=5.0)
var_expense = st.sidebar.slider("Expense & Commission Ratio (%)", 5, 30, 18) / 100.0
profit_margin = st.sidebar.slider("Target Profit Margin (%)", 1, 15, 5) / 100.0
contingency_margin = st.sidebar.slider("Tail Risk Contingency (%)", 0, 15, 5) / 100.0

pricing_engine.fixed_expense = fixed_fee
pricing_engine.variable_expense_ratio = var_expense
pricing_engine.profit_margin = profit_margin
pricing_engine.contingency_margin = contingency_margin


# -------------------------------------------------------------
# MODULE 1: SINGLE QUOTE UNDERWRITING ENGINE
# -------------------------------------------------------------
if nav_mode == "🎯 Underwriting & Quote Engine":
    st.markdown('<div class="main-header">Insurance Risk Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Two-Part Risk Rating Engine: Claim Frequency, Severity & Transparent SHAP Attribution</div>', unsafe_allow_html=True)
    
    col_input, col_results = st.columns([1.1, 1.3], gap="large")
    
    with col_input:
        st.subheader("📋 Policyholder & Vehicle Profile")
        
        with st.expander("👤 Driver Characteristics", expanded=True):
            driv_age = st.slider("Driver Age (Years)", min_value=18, max_value=85, value=35, help="Driver age in years.")
            bonus_malus = st.slider("Bonus-Malus Rating (French MTPL scale)", min_value=50, max_value=230, value=65, 
                                    help="50 = Maximum 50% discount; 100 = Neutral base; >100 = Malus/Penalty surcharge")
        
        with st.expander("🚗 Vehicle Characteristics", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                veh_age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=30, value=4)
                veh_power = st.slider("Vehicle Fiscal Power (CV)", min_value=4, max_value=15, value=6)
            with c2:
                veh_gas = st.selectbox("Fuel Type", ["Regular", "Diesel"], index=0)
                brand_options = ["B1", "B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14"]
                veh_brand = st.selectbox("Vehicle Brand Category", brand_options, index=8)
                
        with st.expander("📍 Geographic & Exposure Context", expanded=True):
            c3, c4 = st.columns(2)
            with c3:
                area_options = ["A", "B", "C", "D", "E", "F"]
                area = st.selectbox("Urban Density Code (A=Rural to F=Urban Core)", area_options, index=3)
                density = st.number_input("Population Density (Inhabitants/km²)", min_value=1, max_value=30000, value=1200)
            with c4:
                regions = [
                    "Centre", "Rhone-Alpes", "Provence-Alpes-Cotes-D'Azur", "Ile-de-France",
                    "Bretagne", "Nord-Pas-de-Calais", "Pays-de-la-Loire", "Languedoc-Roussillon",
                    "Aquitaine", "Poitou-Charentes", "Midi-Pyrenees", "Basse-Normandie",
                    "Bourgogne", "Haute-Normandie", "Picardie", "Auvergne", "Limousin",
                    "Corse", "Champagne-Ardenne", "Alsace", "Franche-Comte"
                ]
                region = st.selectbox("Geographic Region", regions, index=3)
                exposure = st.slider("Coverage Duration (Years of Exposure)", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
                
        # Format single instance dataframe
        single_df = pd.DataFrame([{
            "VehPower": veh_power,
            "VehAge": veh_age,
            "DrivAge": driv_age,
            "BonusMalus": bonus_malus,
            "Density": density,
            "VehBrand": veh_brand,
            "VehGas": veh_gas,
            "Area": area,
            "Region": region,
            "Exposure": exposure
        }])
        
    with col_results:
        st.subheader("💡 Underwriting Decision & Actuarial Premium")
        
        # Inference
        X_trans = preprocessor.transform(single_df)
        claim_prob = float(freq_model.predict_proba(X_trans)[0, 1])
        
        # Severity prediction (log scale exponentiated)
        pred_log_sev = float(sev_model.predict(X_trans)[0])
        pred_sev = float(np.exp(pred_log_sev))
        
        # Adjust for exposure duration
        claim_prob_exp = claim_prob * exposure
        pure_prem = pricing_engine.compute_pure_premium(claim_prob_exp, pred_sev)
        comm_prem = pricing_engine.compute_commercial_premium(pure_prem)
        tier_label, tier_color = pricing_engine.assign_risk_tier(claim_prob_exp)
        
        badge_class = {
            "green": "badge-green",
            "blue": "badge-blue",
            "orange": "badge-orange",
            "red": "badge-red"
        }.get(tier_color, "badge-blue")
        
        st.markdown(f'<span class="{badge_class}">Risk Tier: {tier_label}</span>', unsafe_allow_html=True)
        st.write("")
        
        r1, r2 = st.columns(2)
        with r1:
            st.metric(
                label="Probability of Claim Occurrence P(N > 0)",
                value=f"{claim_prob_exp * 100:.2f}%",
                delta=f"{(claim_prob_exp - 0.0502)*100:+.2f}% vs Portfolio Avg",
                delta_color="inverse"
            )
            st.metric(
                label="Expected Pure Premium (Pure Loss Cost)",
                value=f"€{pure_prem:.2f}",
                help="Pure Premium = P(Claim) × Estimated Severity"
            )
        with r2:
            st.metric(
                label="Estimated Claim Severity E[Y|Claim]",
                value=f"€{pred_sev:.2f}",
                help="Estimated average payout in the event of an at-fault liability incident."
            )
            st.metric(
                label="Recommended Commercial Quote",
                value=f"€{comm_prem:.2f}",
                help="Loaded premium including acquisition, underwriting expenses, profit target, and solvency contingency."
            )
            
        st.markdown("---")
        st.subheader("🔍 Transparent Explainability (SHAP Risk Attribution)")
        st.write("Why was this rate calculated? Here are the top factors impacting this policy's frequency risk:")
        
        shap_factors = explainer.explain_instance(X_trans[0], top_n=6)
        
        # Plotly / Matplotlib bar chart for SHAP
        fig, ax = plt.subplots(figsize=(7, 3.2))
        features = [item["feature"] for item in reversed(shap_factors)]
        impacts = [item["shap_impact"] for item in reversed(shap_factors)]
        colors = ["#EF4444" if val > 0 else "#10B981" for val in impacts]
        
        ax.barh(features, impacts, color=colors)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.6)
        ax.set_xlabel("SHAP Impact on Log-Odds of Claim Occurrence")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Human-readable breakdown table
        shap_display = pd.DataFrame(shap_factors)
        shap_display = shap_display.rename(columns={
            "feature": "Risk Factor",
            "direction": "Risk Effect",
            "shap_impact": "SHAP Contribution",
            "feature_value": "Input Value"
        })
        st.dataframe(shap_display[["Risk Factor", "Risk Effect", "SHAP Contribution"]], use_container_width=True)


# -------------------------------------------------------------
# MODULE 2: PORTFOLIO RISK ANALYTICS
# -------------------------------------------------------------
elif nav_mode == "📊 Portfolio Risk Analytics":
    st.markdown('<div class="main-header">Portfolio Risk & Actuarial Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive Dashboard: Risk Concentration, Lorenz Curve, and Segment Profitability</div>', unsafe_allow_html=True)
    
    # Portfolio KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Policies Evaluated", f"{metadata['total_records']:,}")
    with k2:
        st.metric("Portfolio Claim Occurrence Rate", f"{metadata['frequency_claim_rate']*100:.2f}%")
    with k3:
        st.metric("Mean Claim Severity", f"€{metadata['mean_claim_severity']:.2f}")
    with k4:
        st.metric("Portfolio Gini Index", f"{metadata['portfolio_gini']:.4f}", help="Actuarial Lorenz curve risk separation power (benchmark 0.20 - 0.28)")
        
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📈 Model Performance Comparison")
        
        freq_perf = pd.DataFrame(metadata["frequency_metrics"]).T[["ROC_AUC", "PR_AUC", "Brier_Score"]]
        st.write("**Frequency Models:**")
        st.dataframe(freq_perf.style.highlight_max(subset=["ROC_AUC", "PR_AUC"], color="#D1FAE5").highlight_min(subset=["Brier_Score"], color="#D1FAE5"), use_container_width=True)
        
        sev_perf = pd.DataFrame(metadata["severity_metrics"]).T[["MAE", "RMSE", "Gamma_Deviance"]]
        st.write("**Severity Models:**")
        st.dataframe(sev_perf.style.highlight_min(subset=["MAE", "RMSE", "Gamma_Deviance"], color="#D1FAE5"), use_container_width=True)
        
    with col_chart2:
        st.subheader("🎯 Pure Premium Distribution")
        
        # Compute sample pure premiums
        sample_features = [
            "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density",
            "VehBrand", "VehGas", "Area", "Region"
        ]
        X_sample = preprocessor.transform(portfolio_sample[sample_features])
        sample_probs = freq_model.predict_proba(X_sample)[:, 1]
        sample_sev = np.exp(sev_model.predict(X_sample))
        sample_pp = sample_probs * sample_sev * portfolio_sample["Exposure"].values
        
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.histplot(sample_pp, bins=30, kde=True, color="#2563EB", ax=ax)
        ax.set_title("Estimated Pure Premium per Policy (€)")
        ax.set_xlabel("Pure Premium (€)")
        ax.set_ylabel("Policy Count")
        ax.set_xlim(0, 300)
        st.pyplot(fig)
        
    st.markdown("---")
    st.subheader("🗺️ Risk Differentiation by Geographic Region")
    
    portfolio_sample["Sample_Pure_Premium"] = sample_pp
    portfolio_sample["Sample_Claim_Prob"] = sample_probs
    
    region_agg = portfolio_sample.groupby("Region").agg(
        Policies=("IDpol", "count"),
        Avg_Claim_Prob=("Sample_Claim_Prob", "mean"),
        Avg_Pure_Premium=("Sample_Pure_Premium", "mean")
    ).reset_index().sort_values("Avg_Pure_Premium", ascending=False)
    
    fig_reg, ax_reg = plt.subplots(figsize=(10, 4))
    sns.barplot(data=region_agg, x="Region", y="Avg_Pure_Premium", palette="viridis", ax=ax_reg)
    plt.xticks(rotation=45, ha="right")
    ax_reg.set_ylabel("Avg Pure Premium (€)")
    ax_reg.set_title("Geographic Expected Loss Rating Across French Regions")
    st.pyplot(fig_reg)


# -------------------------------------------------------------
# MODULE 3: BATCH APPLICANT SCORING
# -------------------------------------------------------------
elif nav_mode == "📁 Batch Applicant Scoring":
    st.markdown('<div class="main-header">Batch Policy Scoring Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a batch CSV file of policy applicants for automated risk tiering and premium quoting</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Applicant CSV", type=["csv"])
    
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        use_sample = st.button("📂 Load Pre-existing Evaluation Cohort (1,000 Policies)")
        
    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif use_sample:
        batch_df = portfolio_sample.copy()
        
    if batch_df is not None:
        st.success(f"Loaded cohort of {len(batch_df)} policies.")
        
        feature_cols = [
            "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density",
            "VehBrand", "VehGas", "Area", "Region"
        ]
        
        with st.spinner("Scoring batch against frequency and severity models..."):
            X_batch = preprocessor.transform(batch_df[feature_cols])
            batch_probs = freq_model.predict_proba(X_batch)[:, 1]
            batch_sev = np.exp(sev_model.predict(X_batch))
            
            exposures = batch_df["Exposure"].values if "Exposure" in batch_df.columns else np.ones(len(batch_df))
            batch_pp = batch_probs * batch_sev * exposures
            batch_comm = [pricing_engine.compute_commercial_premium(pp) for pp in batch_pp]
            batch_tiers = [pricing_engine.assign_risk_tier(p)[0] for p in batch_probs]
            
            scored_df = batch_df.copy()
            scored_df["Claim_Probability"] = np.round(batch_probs, 4)
            scored_df["Estimated_Severity_EUR"] = np.round(batch_sev, 2)
            scored_df["Pure_Premium_EUR"] = np.round(batch_pp, 2)
            scored_df["Commercial_Quote_EUR"] = np.round(batch_comm, 2)
            scored_df["Risk_Tier"] = batch_tiers
            
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("Total Cohort Exposure", f"{exposures.sum():,.1f} Policy-Years")
        with b2:
            st.metric("Expected Total Losses", f"€{batch_pp.sum():,.2f}")
        with b3:
            st.metric("Aggregate Target Revenue", f"€{sum(batch_comm):,.2f}")
        with b4:
            st.metric("Underwriting Margin", f"€{(sum(batch_comm) - batch_pp.sum()):,.2f}")
            
        st.subheader("📋 Scored Applicant Register")
        
        # Filter by tier
        tier_filter = st.multiselect(
            "Filter by Risk Tier:",
            options=list(set(batch_tiers)),
            default=list(set(batch_tiers))
        )
        filtered_view = scored_df[scored_df["Risk_Tier"].isin(tier_filter)]
        
        display_cols = [
            "IDpol", "DrivAge", "VehPower", "BonusMalus", "Area", "Region",
            "Claim_Probability", "Estimated_Severity_EUR", "Pure_Premium_EUR",
            "Commercial_Quote_EUR", "Risk_Tier"
        ]
        present_cols = [c for c in display_cols if c in filtered_view.columns]
        st.dataframe(filtered_view[present_cols].head(100), use_container_width=True)
        
        # Download button
        csv_data = scored_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Scored Batch CSV",
            data=csv_data,
            file_name="underwriting_scored_policies.csv",
            mime="text/csv"
        )


# -------------------------------------------------------------
# MODULE 4: ACTUARIAL METHODOLOGY & FORMULAS
# -------------------------------------------------------------
elif nav_mode == "📖 Actuarial Methodology & Formulas":
    st.markdown('<div class="main-header">Actuarial Methodology & Mathematical Foundations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Mathematical Specifications for Two-Part Frequency-Severity Modeling and Risk Pricing</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 1. Two-Part Compound Risk Model
    In actuarial science, aggregate policy claim loss $S$ over an exposure period $e$ is modeled as a compound process:
    $$S = \\sum_{i=1}^{N} Y_i$$
    where:
    - $N \\in \\{0, 1, 2, \\dots\\}$ is the number of claims (Claim Frequency).
    - $Y_i > 0$ represents the monetary loss of the $i$-th claim (Claim Severity).
    
    Assuming independence between frequency $N$ and severity $Y$:
    $$\\mathbb{E}[S] = \\mathbb{E}[N] \\cdot \\mathbb{E}[Y]$$
    
    ### 2. Frequency Modeling (Problem A)
    - **Binary Occurrence Formulation:**
      $$P(N > 0 \\mid \\mathbf{x}) = \\frac{1}{1 + e^{-\\mathbf{x}^\\top \\boldsymbol{\\beta}}}$$
      Trained using Binary Cross-Entropy / Log Loss with LightGBM and XGBoost, calibrated for class imbalance (5.02% base claim rate).
      
    - **Poisson Generalized Linear Model:**
      $$N_i \\sim \\text{Poisson}(\\lambda_i \\cdot e_i), \\quad \\log(\\lambda_i) = \\mathbf{x}_i^\\top \\boldsymbol{\\beta}$$
      where $e_i$ is policy exposure (time at risk).
      
    ### 3. Severity Modeling (Problem B)
    Conditioned on claim occurrence ($N > 0$), severity $Y$ follows a heavy-tailed positive distribution.
    - **Gamma GLM with Log Link:**
      $$Y_i \\sim \\text{Gamma}(\\alpha, \\mu_i), \\quad \\log(\\mu_i) = \\mathbf{x}_i^\\top \\boldsymbol{\\gamma}$$
      Gamma unit deviance loss:
      $$d(y, \\hat{y}) = 2 \\left[ -\\log\\left(\\frac{y}{\\hat{y}}\\right) + \\frac{y - \\hat{y}}{\\hat{y}} \\right]$$
      
    ### 4. Actuarial Commercial Pricing Structure
    To convert expected Pure Premium $\\mathbb{E}[S]$ into a commercial tariff that guarantees solvency and profitability:
    $$\\text{Gross Commercial Premium} = \\left( \\frac{\\text{Pure Premium} + F}{1 - (v + \\pi)} \\right) \\times (1 + c)$$
    where:
    - $F$: Fixed policy administration fee (€35)
    - $v$: Variable operational and acquisition expense ratio (18%)
    - $\\pi$: Underwriting target profit margin (5%)
    - $c$: Solvency contingency capital loading (5%)
    
    ### 5. Regulatory Compliance & Interpretability
    Under Solvency II, IFRS 17, and algorithmic fairness guidelines, black-box insurance pricing is prohibited.
    We deploy **SHAP (SHapley Additive exPlanations)** based on cooperative game theory:
    $$f(\\mathbf{x}) = \\phi_0 + \\sum_{j=1}^{M} \\phi_j(\\mathbf{x})$$
    ensuring every policyholder's rate has transparent, defensible attribution.
    """)

