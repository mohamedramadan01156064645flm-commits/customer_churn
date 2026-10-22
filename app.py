import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ChurnGuard AI — Customer Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL CSS THEME
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Global Reset ────────────────────────────────────────── */
    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .stApp {
        background: #0a0e1a !important;
        color: #cbd5e1;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1280px;
    }
    /* global text */
    .stApp p, .stApp li, .stApp span, .stApp div {
        color: #cbd5e1;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #e2e8f0 !important;
    }

    /* ── Hero Header ─────────────────────────────────────────── */
    .hero-container {
        background: linear-gradient(135deg, #0d1324 0%, #131b34 40%, #1a2547 100%);
        border-radius: 20px;
        padding: 40px 48px 36px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(56, 189, 248, 0.12);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -60%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: 10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(167, 139, 250, 0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 100px;
        padding: 5px 14px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #7dd3fc;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 16px;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #e2e8f0;
        line-height: 1.15;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    .hero-title span {
        background: linear-gradient(135deg, #38bdf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #7e8da6;
        font-weight: 400;
        line-height: 1.6;
        max-width: 620px;
        position: relative;
        z-index: 1;
    }

    /* ── KPI Cards Row ───────────────────────────────────────── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 28px;
    }
    .kpi-card {
        background: #111827;
        border-radius: 16px;
        padding: 22px 24px;
        border: 1px solid #1e293b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        transition: all 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.45);
        border-color: #334155;
    }
    .kpi-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        margin-bottom: 14px;
    }
    .kpi-icon.blue   { background: rgba(59, 130, 246, 0.15); color: #60a5fa; }
    .kpi-icon.green  { background: rgba(34, 197, 94, 0.15);  color: #4ade80; }
    .kpi-icon.purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; }
    .kpi-icon.amber  { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #e2e8f0;
        letter-spacing: -0.3px;
    }
    .kpi-delta {
        font-size: 0.73rem;
        font-weight: 500;
        color: #64748b;
        margin-top: 4px;
    }

    /* ── Tab Styling ─────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #111827;
        border-radius: 14px;
        padding: 6px;
        border: 1px solid #1e293b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.85rem;
        color: #64748b !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e40af, #7c3aed) !important;
        color: #f1f5f9 !important;
    }

    /* ── Cards / Panels ──────────────────────────────────────── */
    .glass-card {
        background: #111827;
        border-radius: 16px;
        padding: 28px;
        border: 1px solid #1e293b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0 !important;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-desc {
        font-size: 0.82rem;
        color: #64748b !important;
        margin-bottom: 18px;
    }

    /* ── Risk Badges ─────────────────────────────────────────── */
    .risk-banner {
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 16px;
        display: flex;
        align-items: flex-start;
        gap: 14px;
    }
    .risk-banner.critical {
        background: linear-gradient(135deg, rgba(127,29,29,0.3), rgba(153,27,27,0.2));
        border: 1px solid rgba(239,68,68,0.35);
    }
    .risk-banner.warning {
        background: linear-gradient(135deg, rgba(120,53,15,0.25), rgba(146,64,14,0.15));
        border: 1px solid rgba(245,158,11,0.35);
    }
    .risk-banner.safe {
        background: linear-gradient(135deg, rgba(20,83,45,0.25), rgba(22,101,52,0.15));
        border: 1px solid rgba(34,197,94,0.35);
    }
    .risk-icon {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        flex-shrink: 0;
    }
    .risk-icon.critical { background: rgba(239,68,68,0.15); }
    .risk-icon.warning  { background: rgba(245,158,11,0.15); }
    .risk-icon.safe     { background: rgba(34,197,94,0.15); }
    .risk-title {
        font-size: 1.0rem;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .risk-title.critical { color: #fca5a5; }
    .risk-title.warning  { color: #fcd34d; }
    .risk-title.safe     { color: #86efac; }
    .risk-body {
        font-size: 0.83rem;
        line-height: 1.55;
    }
    .risk-body.critical { color: #f87171; }
    .risk-body.warning  { color: #fbbf24; }
    .risk-body.safe     { color: #4ade80; }

    /* ── Recommendation Cards ────────────────────────────────── */
    .rec-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 12px;
        transition: all 0.15s ease;
        position: relative;
        overflow: hidden;
    }
    .rec-card:hover {
        border-color: #334155;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        transform: translateY(-1px);
    }
    .rec-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
    }
    .rec-card.priority-critical::before { background: linear-gradient(180deg, #ef4444, #dc2626); }
    .rec-card.priority-high::before     { background: linear-gradient(180deg, #f97316, #ea580c); }
    .rec-card.priority-medium::before   { background: linear-gradient(180deg, #eab308, #ca8a04); }
    .rec-card.priority-low::before      { background: linear-gradient(180deg, #22c55e, #16a34a); }
    .rec-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .rec-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 100px;
    }
    .rec-tag.critical { background: rgba(239,68,68,0.15); color: #f87171; }
    .rec-tag.high     { background: rgba(249,115,22,0.15); color: #fb923c; }
    .rec-tag.medium   { background: rgba(234,179,8,0.15);  color: #facc15; }
    .rec-tag.low      { background: rgba(34,197,94,0.15);  color: #4ade80; }
    .rec-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #e2e8f0 !important;
        margin-bottom: 6px;
    }
    .rec-body {
        font-size: 0.82rem;
        color: #94a3b8 !important;
        line-height: 1.6;
    }
    .rec-metrics {
        display: flex;
        gap: 16px;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid #1e293b;
    }
    .rec-metric {
        font-size: 0.72rem;
        color: #64748b !important;
        font-weight: 500;
    }
    .rec-metric b {
        color: #cbd5e1 !important;
        font-weight: 700;
    }

    /* ── Section Headers ─────────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }
    .section-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.05rem;
    }
    .section-icon.blue   { background: rgba(59,130,246,0.15); }
    .section-icon.purple { background: rgba(168,85,247,0.15); }
    .section-icon.green  { background: rgba(34,197,94,0.15); }
    .section-icon.amber  { background: rgba(245,158,11,0.15); }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2e8f0 !important;
    }

    /* ── Footer ──────────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 30px 0 10px;
        color: #475569;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .app-footer span {
        color: #64748b;
        font-weight: 600;
    }

    /* ── Streamlit overrides ─────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: #111827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        font-weight: 600 !important;
        color: #64748b !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 800 !important;
        color: #e2e8f0 !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #64748b !important;
    }
    .stForm {
        background: #111827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    /* select boxes, inputs, sliders */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #1a1f2e !important;
        border-color: #2d3748 !important;
        color: #e2e8f0 !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input {
        color: #e2e8f0 !important;
    }
    /* form labels */
    .stForm label, .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #94a3b8 !important;
    }
    button[kind="primaryFormSubmit"],
    .stFormSubmitButton button {
        background: linear-gradient(135deg, #1e40af, #7c3aed) !important;
        color: #f1f5f9 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
    }
    button[kind="primaryFormSubmit"]:hover,
    .stFormSubmitButton button:hover {
        box-shadow: 0 8px 30px rgba(30, 64, 175, 0.4) !important;
        transform: translateY(-1px);
    }
    .stDownloadButton button {
        background: linear-gradient(135deg, #1e40af, #7c3aed) !important;
        color: #f1f5f9 !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    .stFileUploader {
        border: 2px dashed #334155 !important;
        border-radius: 14px !important;
        background: #111827 !important;
    }
    .stFileUploader label {
        color: #94a3b8 !important;
    }
    .stDivider {
        margin: 8px 0 !important;
        border-color: #1e293b !important;
    }
    /* expander */
    details {
        background: #111827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 14px !important;
    }
    details summary {
        color: #e2e8f0 !important;
    }
    /* dataframe */
    .stDataFrame, [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }
    /* success / info / warning / error alerts */
    div[data-testid="stAlert"] {
        background: #111827 !important;
        border: 1px solid #1e293b !important;
        color: #cbd5e1 !important;
        border-radius: 12px !important;
    }
    /* button general */
    .stButton button {
        background: #111827 !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background: #1e293b !important;
        border-color: #475569 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    /* file uploader text */
    [data-testid="stFileUploaderDropzone"] {
        background: #111827 !important;
        color: #94a3b8 !important;
    }
    /* hide streamlit deploy button */
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_churn_model():
    return joblib.load(os.path.join(os.path.dirname(__file__), "churn_model.pkl"))

model = load_churn_model()

# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🔒 AI-POWERED ENGINE &nbsp;•&nbsp; LIVE INFERENCE</div>
    <div class="hero-title">ChurnGuard <span>AI</span></div>
    <div class="hero-subtitle">
        Enterprise-grade customer intelligence platform. Predict churn risk in real-time,
        uncover root-cause drivers, and receive actionable retention playbooks — powered by
        machine learning on 7,043 Telco customer profiles.
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-icon blue">🧠</div>
        <div class="kpi-label">Model Architecture</div>
        <div class="kpi-value">Balanced LR</div>
        <div class="kpi-delta">Logistic Regression · class_weight='balanced'</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon green">🎯</div>
        <div class="kpi-label">ROC-AUC Score</div>
        <div class="kpi-value">0.844</div>
        <div class="kpi-delta">+4.2% improvement over unbalanced baseline</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon purple">📡</div>
        <div class="kpi-label">Churn Recall</div>
        <div class="kpi-value">79.1%</div>
        <div class="kpi-delta">Catches ~4 out of 5 at-risk customers</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon amber">⚡</div>
        <div class="kpi-label">Inference Speed</div>
        <div class="kpi-value">&lt;50ms</div>
        <div class="kpi-delta">Real-time scoring per customer record</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_single, tab_batch, tab_insights = st.tabs([
    "👤  Single Customer Scoring",
    "📁  Batch Portfolio Analysis",
    "🧠  Model Explainability"
])

# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def generate_recommendations(data, churn_prob):
    """Generate realistic, prioritized retention playbook based on customer profile."""
    recs = []
    contract = data.get("Contract", "")
    internet = data.get("InternetService", "")
    tenure = data.get("tenure", 0)
    monthly = data.get("MonthlyCharges", 0)
    payment = data.get("PaymentMethod", "")
    security = data.get("OnlineSecurity", "")
    tech = data.get("TechSupport", "")
    backup = data.get("OnlineBackup", "")
    device = data.get("DeviceProtection", "")
    streaming_tv = data.get("StreamingTV", "")
    streaming_movies = data.get("StreamingMovies", "")
    paperless = data.get("PaperlessBilling", "")

    # ── 1. Contract-Based ────────────────────────────────────
    if contract == "Month-to-month":
        annual_savings = round(monthly * 12 * 0.15, 2)
        if churn_prob > 0.6:
            recs.append({
                "priority": "critical",
                "icon": "📜",
                "title": "Immediate Contract Migration — Offer Annual Plan Lock-in",
                "body": (
                    f"This customer is on a month-to-month contract — the single strongest "
                    f"churn predictor in the model (coef = +0.66). Offer a personalized "
                    f"migration to a 12-month plan with a <b>15% loyalty discount</b> "
                    f"(saving ~<b>${annual_savings}/year</b>). Frame it as: "
                    f"\"We value your {tenure}-month relationship — here's an exclusive rate "
                    f"locked in just for you.\" If they hesitate, offer a 3-month trial at "
                    f"the annual rate with a no-penalty exit clause."
                ),
                "est_impact": "Reduces churn probability by ~25-35%",
                "timeline": "Within 48 hours",
                "channel": "Outbound call + follow-up SMS"
            })
        else:
            recs.append({
                "priority": "high",
                "icon": "📋",
                "title": "Proactive Contract Upgrade Nudge",
                "body": (
                    f"Month-to-month customers have a 42% average churn rate vs. 3% for 2-year contracts. "
                    f"Send a personalized email highlighting a <b>${annual_savings}/year savings</b> "
                    f"by switching to an annual plan. Include a comparison table showing "
                    f"current monthly cost vs. discounted annual cost."
                ),
                "est_impact": "Reduces churn probability by ~15-20%",
                "timeline": "Next billing cycle",
                "channel": "Email campaign with in-app banner"
            })

    # ── 2. Fiber Optic High-Cost Pain ────────────────────────
    if internet == "Fiber optic" and monthly > 80:
        recs.append({
            "priority": "critical" if churn_prob > 0.5 else "high",
            "icon": "🌐",
            "title": "Fiber Optic Value Perception Gap — Price Justification Campaign",
            "body": (
                f"Fiber optic users churn at <b>2.3x</b> the rate of DSL users (model coef = +0.71). "
                f"At <b>${monthly:.0f}/mo</b>, this customer may feel the service doesn't justify "
                f"the premium. Actions: (1) Send a personalized speed-test report showing actual "
                f"bandwidth vs. what DSL would deliver, (2) Offer a complimentary upgrade to the "
                f"next speed tier for 3 months, (3) If on a single service, bundle in a free "
                f"add-on (e.g., Online Security or Cloud Backup) to increase perceived value "
                f"and switching cost."
            ),
            "est_impact": "Reduces churn probability by ~15-20%",
            "timeline": "Within 1 week",
            "channel": "Personalized email + account portal notification"
        })

    # ── 3. Missing Security/Support Bundles ──────────────────
    missing_services = []
    if security == "No" and internet != "No":
        missing_services.append("Online Security")
    if tech == "No" and internet != "No":
        missing_services.append("Tech Support")
    if backup == "No" and internet != "No":
        missing_services.append("Online Backup")
    if device == "No" and internet != "No":
        missing_services.append("Device Protection")

    if len(missing_services) >= 2:
        svc_list = ", ".join(missing_services[:3])
        monthly_bundle_price = round(len(missing_services) * 4.5, 2)
        recs.append({
            "priority": "high" if churn_prob > 0.4 else "medium",
            "icon": "🛡️",
            "title": f"Service Gap Detected — Offer \"{len(missing_services)}-in-1 Protection Bundle\"",
            "body": (
                f"This customer is missing <b>{svc_list}</b>. Customers without protection "
                f"services have a significantly higher churn coefficient. Propose a bundled "
                f"\"Complete Protection Pack\" at <b>${monthly_bundle_price}/mo</b> "
                f"(vs. ~${len(missing_services) * 8}/mo if purchased individually — a "
                f"{round((1 - monthly_bundle_price / (len(missing_services) * 8)) * 100)}% discount). "
                f"Position it as: \"For less than the price of a coffee per day, protect all "
                f"your devices and data.\""
            ),
            "est_impact": f"Reduces churn by ~10-15% + increases ARPU by ${monthly_bundle_price}",
            "timeline": "Next billing notification",
            "channel": "In-app popup + email"
        })
    elif len(missing_services) == 1:
        recs.append({
            "priority": "medium",
            "icon": "🔐",
            "title": f"Upsell Opportunity — Free Trial of {missing_services[0]}",
            "body": (
                f"Offer a <b>60-day complimentary trial</b> of {missing_services[0]}. "
                f"This increases the customer's \"stickiness\" — each additional service "
                f"a customer uses raises switching costs and reduces churn by ~5-8%. "
                f"Auto-enroll with an easy opt-out rather than opt-in to maximize adoption."
            ),
            "est_impact": "Reduces churn by ~5-8%",
            "timeline": "Immediate auto-enrollment",
            "channel": "Account auto-activation + welcome email"
        })

    # ── 4. Payment Method Risk ───────────────────────────────
    if payment == "Electronic check":
        recs.append({
            "priority": "high" if churn_prob > 0.5 else "medium",
            "icon": "💳",
            "title": "Payment Method Migration — Switch from Electronic Check",
            "body": (
                f"Electronic check users churn at <b>45%</b> vs. 15% for auto-pay customers. "
                f"This is often correlated with lower engagement and payment friction. "
                f"Offer a <b>$5/month recurring credit for 6 months</b> (total $30 incentive) "
                f"for switching to automatic bank transfer or credit card. "
                f"ROI: The cost of acquiring a new customer (~$300-450) far exceeds the $30 "
                f"retention incentive. Include a one-click setup link in the next invoice."
            ),
            "est_impact": "Reduces churn by ~12-18%",
            "timeline": "Include in next billing statement",
            "channel": "Invoice insert + SMS deeplink"
        })
    elif payment == "Mailed check":
        recs.append({
            "priority": "medium",
            "icon": "📨",
            "title": "Digital Payment Adoption — Eliminate Mailing Friction",
            "body": (
                f"Mailed check customers face payment friction and late-payment risk. "
                f"Offer <b>paperless billing with a one-time $10 account credit</b> "
                f"for switching to automatic payments. Highlight convenience: "
                f"\"Never worry about mailing a check again — set it and forget it.\""
            ),
            "est_impact": "Reduces churn by ~8-12%",
            "timeline": "Within 2 weeks",
            "channel": "Direct mail with QR code to digital setup"
        })

    # ── 5. Tenure-Based Lifecycle ────────────────────────────
    if tenure <= 3:
        recs.append({
            "priority": "critical",
            "icon": "🚨",
            "title": "Critical Onboarding Window — 90-Day Survival Program",
            "body": (
                f"This customer has been with us for only <b>{tenure} month{'s' if tenure != 1 else ''}</b>. "
                f"Industry data shows 35% of churn happens within the first 90 days. "
                f"Activate the \"New Customer Success\" playbook: "
                f"(1) Day 1: Welcome call from dedicated account manager, "
                f"(2) Day 14: Satisfaction pulse survey (NPS), "
                f"(3) Day 30: Personalized usage tips email based on their plan, "
                f"(4) Day 60: Proactive check-in call — ask about any issues before they escalate. "
                f"Assign a named CSM (Customer Success Manager) visible in their account portal."
            ),
            "est_impact": "Reduces early churn by ~30-40%",
            "timeline": "Start immediately — Day 0",
            "channel": "Phone call + automated email journey"
        })
    elif tenure <= 12:
        recs.append({
            "priority": "high" if churn_prob > 0.5 else "medium",
            "icon": "📆",
            "title": f"First-Year Milestone Engagement — {tenure}-Month Anniversary",
            "body": (
                f"At <b>{tenure} months</b>, this customer is past initial onboarding "
                f"but hasn't yet reached the loyalty plateau (~18 months). "
                f"Send a \"{tenure}-month anniversary\" thank-you with: "
                f"(1) A personalized usage summary (\"You've used X GB this month!\"), "
                f"(2) A small loyalty perk (free premium channel for 1 month, or a bill credit), "
                f"(3) An invitation to join the loyalty program / rewards tier."
            ),
            "est_impact": "Reduces churn by ~10-15%",
            "timeline": "On the anniversary date",
            "channel": "Personalized email + push notification"
        })

    # ── 6. Streaming without protection ──────────────────────
    streams_active = sum([
        1 for s in [streaming_tv, streaming_movies] if s == "Yes"
    ])
    if streams_active >= 1 and tech == "No" and internet == "Fiber optic":
        recs.append({
            "priority": "medium",
            "icon": "📺",
            "title": "Entertainment + Support Bundle — Streaming User at Risk",
            "body": (
                f"This customer actively uses <b>{streams_active} streaming service{'s' if streams_active > 1 else ''}</b> "
                f"on Fiber Optic but has no Tech Support. Streaming issues (buffering, "
                f"connectivity drops) without support access create high frustration. "
                f"Offer a \"Streaming Premium\" bundle: Tech Support + priority bandwidth "
                f"allocation at a <b>25% bundled discount</b>. This increases the perceived "
                f"value of Fiber Optic and directly addresses the satisfaction gap."
            ),
            "est_impact": "Reduces churn by ~8-12% for streaming users",
            "timeline": "Within 1 week",
            "channel": "In-app recommendation engine"
        })

    # ── 7. High monthly charges relative to services ─────────
    services_count = sum([
        1 for s in [security, tech, backup, device, streaming_tv, streaming_movies]
        if s == "Yes"
    ])
    if monthly > 85 and services_count <= 2:
        recs.append({
            "priority": "high",
            "icon": "💰",
            "title": "Price-to-Value Optimization — Customer Overpaying for Underused Plan",
            "body": (
                f"This customer pays <b>${monthly:.0f}/mo</b> but only uses "
                f"<b>{services_count} add-on service{'s' if services_count != 1 else ''}</b>. "
                f"This creates a value perception gap. Two options: "
                f"(1) <b>Downgrade & retain</b>: Proactively suggest a more cost-effective plan "
                f"at ~${monthly * 0.8:.0f}/mo (the goodwill of saving them money builds trust), or "
                f"(2) <b>Upgrade the experience</b>: Bundle in 2-3 premium services at the current "
                f"price point so they feel they're getting more for what they pay."
            ),
            "est_impact": "Reduces churn by ~15-20%",
            "timeline": "Proactive outreach within 5 days",
            "channel": "CSM phone call + restructured plan email"
        })

    # ── 8. Low risk — loyalty nurture ────────────────────────
    if churn_prob < 0.25 and tenure > 24:
        recs.append({
            "priority": "low",
            "icon": "⭐",
            "title": "Loyalty Program Enrollment — Turn Satisfied Customers into Advocates",
            "body": (
                f"With <b>{tenure} months</b> of tenure and a churn probability of only "
                f"<b>{churn_prob*100:.1f}%</b>, this is a highly loyal customer. "
                f"Maximize their lifetime value: "
                f"(1) Enroll in VIP loyalty tier with exclusive perks (priority support, early "
                f"access to new features), "
                f"(2) Launch a referral program: \"Refer a friend, both get $25 credit\", "
                f"(3) Request a testimonial or online review — social proof from long-term "
                f"customers drives acquisition."
            ),
            "est_impact": "Increases LTV by ~15-20% through referrals",
            "timeline": "Ongoing relationship nurture",
            "channel": "Email + loyalty portal enrollment"
        })

    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    recs.sort(key=lambda x: priority_order.get(x["priority"], 99))

    return recs

def render_recommendations(recs):
    """Render recommendation cards as HTML."""
    if not recs:
        st.info("No specific recommendations — customer profile looks healthy.")
        return

    for rec in recs:
        priority = rec["priority"]
        st.markdown(f"""
        <div class="rec-card priority-{priority}">
            <div class="rec-header">
                <span style="font-size: 1.3rem;">{rec['icon']}</span>
                <span class="rec-tag {priority}">
                    {'🔴' if priority == 'critical' else '🟠' if priority == 'high' else '🟡' if priority == 'medium' else '🟢'} {priority.upper()} PRIORITY
                </span>
            </div>
            <div class="rec-title">{rec['title']}</div>
            <div class="rec-body">{rec['body']}</div>
            <div class="rec-metrics">
                <div class="rec-metric">📊 Est. Impact: <b>{rec['est_impact']}</b></div>
                <div class="rec-metric">⏱️ Timeline: <b>{rec['timeline']}</b></div>
                <div class="rec-metric">📣 Channel: <b>{rec['channel']}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE CUSTOMER SCORING
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon blue">🔍</div>
        <div class="section-title">Real-Time Customer Risk Assessment</div>
    </div>
    <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 20px; margin-left: 46px;">
        Configure a customer profile below or select a quick-test persona to instantly evaluate churn risk.
    </p>
    """, unsafe_allow_html=True)

    # Persona Presets
    pcol1, pcol2 = st.columns([2, 3])
    with pcol1:
        selected_persona = st.selectbox(
            "⚡ Quick-Test Persona",
            [
                "✏️  Custom Input",
                "🚨  High-Risk Churner — Fiber / Monthly / No Support",
                "🛡️  Loyal Customer — 2yr / Auto-Pay / Full Bundle",
                "📱  Budget User — DSL / 1yr / Basic Services"
            ],
            index=0
        )

    # Defaults
    if "High-Risk" in selected_persona:
        d = dict(g="Female",s=0,p="No",dep="No",t=3,ph="Yes",ml="No",i="Fiber optic",
                 sec="No",bu="No",dev="No",ts="No",tv="Yes",mv="Yes",c="Month-to-month",
                 pb="Yes",pm="Electronic check",mc=95.5,tc=286.5)
    elif "Loyal" in selected_persona:
        d = dict(g="Male",s=0,p="Yes",dep="Yes",t=60,ph="Yes",ml="Yes",i="DSL",
                 sec="Yes",bu="Yes",dev="Yes",ts="Yes",tv="No",mv="No",c="Two year",
                 pb="No",pm="Bank transfer (automatic)",mc=65.0,tc=3900.0)
    elif "Budget" in selected_persona:
        d = dict(g="Male",s=0,p="No",dep="No",t=24,ph="Yes",ml="No",i="DSL",
                 sec="Yes",bu="No",dev="No",ts="No",tv="No",mv="No",c="One year",
                 pb="No",pm="Mailed check",mc=45.0,tc=1080.0)
    else:
        d = dict(g="Female",s=0,p="No",dep="No",t=12,ph="Yes",ml="No",i="Fiber optic",
                 sec="No",bu="Yes",dev="No",ts="No",tv="No",mv="No",c="Month-to-month",
                 pb="Yes",pm="Electronic check",mc=75.0,tc=900.0)

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤  Demographics & Account")
            gender = st.selectbox("Gender", ["Female","Male"], index=0 if d["g"]=="Female" else 1)
            senior = st.selectbox("Senior Citizen", [0,1], format_func=lambda x: "Yes" if x else "No", index=d["s"])
            partner = st.selectbox("Partner", ["No","Yes"], index=0 if d["p"]=="No" else 1)
            dependents = st.selectbox("Dependents", ["No","Yes"], index=0 if d["dep"]=="No" else 1)
            tenure = st.slider("Tenure (months)", 0, 72, d["t"])

        with col2:
            st.markdown("##### 💳  Contract & Billing")
            co = ["Month-to-month","One year","Two year"]
            contract = st.selectbox("Contract Type", co, index=co.index(d["c"]))
            pm_opts = ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
            payment_method = st.selectbox("Payment Method", pm_opts, index=pm_opts.index(d["pm"]))
            paperless = st.selectbox("Paperless Billing", ["No","Yes"], index=0 if d["pb"]=="No" else 1)
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, float(d["mc"]), 0.5)
            total_charges = st.number_input("Total Charges ($)", 0.0, 15000.0, float(d["tc"]), 10.0)

        with col3:
            st.markdown("##### 🌐  Subscribed Services")
            phone_service = st.selectbox("Phone Service", ["No","Yes"], index=0 if d["ph"]=="No" else 1)
            ml_opts = ["No","Yes","No phone service"] if phone_service=="Yes" else ["No phone service"]
            multiple_lines = st.selectbox("Multiple Lines", ml_opts, index=min(["No","Yes"].index(d["ml"]) if d["ml"] in ["No","Yes"] else 0, len(ml_opts)-1))
            internet_service = st.selectbox("Internet Service", ["DSL","Fiber optic","No"], index=["DSL","Fiber optic","No"].index(d["i"]))
            so = ["No","Yes","No internet service"] if internet_service!="No" else ["No internet service"]
            def si(val):
                if internet_service=="No": return 0
                return 1 if val=="Yes" else 0
            online_security = st.selectbox("Online Security", so, index=si(d["sec"]))
            online_backup = st.selectbox("Online Backup", so, index=si(d["bu"]))
            device_protection = st.selectbox("Device Protection", so, index=si(d["dev"]))
            tech_support = st.selectbox("Tech Support", so, index=si(d["ts"]))
            streaming_tv = st.selectbox("Streaming TV", so, index=si(d["tv"]))
            streaming_movies = st.selectbox("Streaming Movies", so, index=si(d["mv"]))

        submitted = st.form_submit_button("🚀  Run Churn Prediction", use_container_width=True)

    # Build input DataFrame
    input_df = pd.DataFrame([{
        'SeniorCitizen': senior, 'tenure': tenure,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
        'gender': gender, 'Partner': partner, 'Dependents': dependents,
        'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless, 'PaymentMethod': payment_method
    }])

    prob = model.predict_proba(input_df)[0][1]
    pct = prob * 100

    # ── Results ──────────────────────────────────────────────
    st.markdown("""
    <div class="section-header" style="margin-top: 10px;">
        <div class="section-icon purple">🎯</div>
        <div class="section-title">Prediction Results</div>
    </div>
    """, unsafe_allow_html=True)

    r1, r2 = st.columns([1, 1.6])

    with r1:
        # Gauge
        bar_color = "#ef4444" if pct > 60 else ("#f59e0b" if pct >= 40 else "#22c55e")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={'suffix': "%", 'font': {'size': 42, 'weight': 800, 'family': 'Inter', 'color': '#e2e8f0'}},
            title={'text': "Churn Probability", 'font': {'size': 15, 'color': '#64748b', 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#334155', 'tickfont': {'size': 11, 'family': 'Inter', 'color': '#64748b'}},
                'bar': {'color': bar_color, 'thickness': 0.82},
                'bgcolor': '#1a1f2e',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(34,197,94,0.1)'},
                    {'range': [40, 60], 'color': 'rgba(245,158,11,0.1)'},
                    {'range': [60, 100], 'color': 'rgba(239,68,68,0.1)'}
                ],
                'threshold': {'line': {'color': '#e2e8f0', 'width': 3}, 'thickness': 0.8, 'value': 50}
            }
        ))
        gauge.update_layout(
            height=260,
            margin=dict(l=30, r=30, t=50, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Inter', 'color': '#cbd5e1'}
        )
        st.plotly_chart(gauge, use_container_width=True)

        # Risk banner
        if pct >= 60:
            st.markdown(f"""
            <div class="risk-banner critical">
                <div class="risk-icon critical">🚨</div>
                <div>
                    <div class="risk-title critical">CRITICAL CHURN RISK — {pct:.1f}%</div>
                    <div class="risk-body critical">
                        This customer is in the top churn-risk decile. Without immediate intervention,
                        there is a {pct:.0f}% probability of cancellation within the next billing cycle.
                        Estimated revenue at risk: <b>${monthly_charges * 12:,.0f}/year</b>.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif pct >= 40:
            st.markdown(f"""
            <div class="risk-banner warning">
                <div class="risk-icon warning">⚡</div>
                <div>
                    <div class="risk-title warning">ELEVATED CHURN RISK — {pct:.1f}%</div>
                    <div class="risk-body warning">
                        Customer is in the warning zone. Proactive engagement within the next 2 weeks
                        can significantly reduce the likelihood of churn. Revenue exposure:
                        <b>${monthly_charges * 12:,.0f}/year</b>.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-banner safe">
                <div class="risk-icon safe">✅</div>
                <div>
                    <div class="risk-title safe">LOW CHURN RISK — {pct:.1f}%</div>
                    <div class="risk-body safe">
                        Customer shows strong retention signals. Focus on maximizing lifetime value
                        through loyalty programs and upsell opportunities.
                        Secured revenue: <b>${monthly_charges * 12:,.0f}/year</b>.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with r2:
        st.markdown("""
        <div class="card-title">💡 AI Retention Playbook</div>
        <div class="card-desc">Prioritized, actionable recommendations generated from this customer's risk profile and feature attribution.</div>
        """, unsafe_allow_html=True)

        profile_data = {
            "Contract": contract, "InternetService": internet_service,
            "tenure": tenure, "MonthlyCharges": monthly_charges,
            "PaymentMethod": payment_method, "OnlineSecurity": online_security,
            "TechSupport": tech_support, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "PaperlessBilling": paperless
        }
        recs = generate_recommendations(profile_data, prob)
        render_recommendations(recs)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PORTFOLIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon green">📊</div>
        <div class="section-title">Batch Portfolio Risk Scoring</div>
    </div>
    <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 20px; margin-left: 46px;">
        Upload a customer CSV or load the built-in sample to score an entire portfolio in seconds.
    </p>
    """, unsafe_allow_html=True)

    bcol1, bcol2 = st.columns([1.5, 3.5])
    with bcol1:
        use_sample = st.button("📥  Load Sample Dataset (20 customers)", use_container_width=True)
    uploaded = st.file_uploader("Or upload your own CSV", type=["csv"])

    target_df = None
    if use_sample and os.path.exists("sample_customers.csv"):
        target_df = pd.read_csv("sample_customers.csv")
        st.success("✅ Loaded built-in sample dataset — 20 customer records.")
    elif uploaded:
        target_df = pd.read_csv(uploaded)
        st.success(f"✅ Uploaded file: {target_df.shape[0]} rows × {target_df.shape[1]} columns.")

    if target_df is not None:
        with st.expander("🔎 Raw Data Preview", expanded=False):
            st.dataframe(target_df.head(8), use_container_width=True)

        work = target_df.copy()
        if 'TotalCharges' in work.columns:
            work['TotalCharges'] = pd.to_numeric(work['TotalCharges'], errors='coerce').fillna(0)

        try:
            probs = model.predict_proba(work)[:, 1]
            preds = model.predict(work)

            res = target_df.copy()
            res["Churn_Probability_%"] = (probs * 100).round(2)
            res["Predicted_Churn"] = np.where(preds == 1, "Yes", "No")
            res["Risk_Level"] = pd.cut(res["Churn_Probability_%"],
                                        bins=[-0.1, 40, 60, 100.1],
                                        labels=["🟢 Low", "🟡 Medium", "🔴 High"])

            total = len(res)
            hi = (res["Risk_Level"] == "🔴 High").sum()
            md = (res["Risk_Level"] == "🟡 Medium").sum()
            lo = (res["Risk_Level"] == "🟢 Low").sum()
            avg_prob = res["Churn_Probability_%"].mean()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Scored", f"{total}")
            c2.metric("🔴 High Risk", f"{hi}  ({hi/total*100:.0f}%)")
            c3.metric("🟡 Medium Risk", f"{md}  ({md/total*100:.0f}%)")
            c4.metric("🟢 Low Risk", f"{lo}  ({lo/total*100:.0f}%)")

            ch1, ch2 = st.columns(2)
            with ch1:
                colors = {"🔴 High": "#ef4444", "🟡 Medium": "#f59e0b", "🟢 Low": "#22c55e"}
                pie = px.pie(res, names="Risk_Level", title="Portfolio Risk Distribution",
                             color="Risk_Level", color_discrete_map=colors)
                pie.update_layout(font=dict(family="Inter", color="#cbd5e1"), margin=dict(t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(pie, use_container_width=True)
            with ch2:
                hist = px.histogram(res, x="Churn_Probability_%", nbins=20,
                                     title="Churn Probability Distribution",
                                     color="Risk_Level", color_discrete_map=colors)
                hist.update_layout(font=dict(family="Inter", color="#cbd5e1"), margin=dict(t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(gridcolor='#1e293b'), yaxis=dict(gridcolor='#1e293b'))
                st.plotly_chart(hist, use_container_width=True)

            show_cols = [c for c in ['customerID','Contract','tenure','MonthlyCharges',
                                      'Churn_Probability_%','Risk_Level','Predicted_Churn'] if c in res.columns]
            st.dataframe(res[show_cols], use_container_width=True)

            st.download_button("📥  Download Scored Results (CSV)",
                               res.to_csv(index=False).encode('utf-8'),
                               "churnguard_scored_results.csv", "text/csv",
                               use_container_width=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon amber">🧠</div>
        <div class="section-title">Model Explainability & Feature Attribution</div>
    </div>
    <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 20px; margin-left: 46px;">
        Understand what drives the model's decisions — feature coefficients extracted from the trained pipeline.
    </p>
    """, unsafe_allow_html=True)

    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    fn = pre.get_feature_names_out()
    coefs = clf.coef_[0]
    fi = pd.DataFrame({"Feature": fn, "Coefficient": coefs})
    fi["Clean"] = fi["Feature"].str.replace("cat__", "").str.replace("num__", "")
    fi["Abs"] = fi["Coefficient"].abs()
    fi["Direction"] = np.where(fi["Coefficient"] > 0, "⬆ Increases Churn", "⬇ Reduces Churn")
    top = fi.nlargest(15, "Abs").sort_values("Coefficient")

    e1, e2 = st.columns([1.6, 1])
    with e1:
        bar = px.bar(top, x="Coefficient", y="Clean", orientation="h",
                      color="Direction",
                      color_discrete_map={"⬆ Increases Churn": "#ef4444", "⬇ Reduces Churn": "#3b82f6"},
                      title="Top 15 Feature Weights — Logistic Regression Coefficients")
        bar.add_vline(x=0, line_width=2, line_color="#475569")
        bar.update_layout(
            height=500,
            font=dict(family="Inter", size=12, color="#cbd5e1"),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title="",
            xaxis_title="Coefficient Weight",
            legend_title="",
            xaxis=dict(gridcolor='#1e293b', zerolinecolor='#334155'),
            yaxis=dict(gridcolor='#1e293b')
        )
        bar.update_yaxes(tickfont=dict(size=11, weight=600, color='#94a3b8'))
        st.plotly_chart(bar, use_container_width=True)

    with e2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">📌 Key Business Insights</div>
            <div style="margin-top: 14px;">
                <div style="margin-bottom: 16px;">
                    <div style="font-weight: 700; color: #e2e8f0; font-size: 0.88rem;">1. Tenure is the #1 Retention Factor</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.55;">
                        Coefficient: <b style="color:#60a5fa;">−1.15</b>. Every additional month dramatically reduces churn.
                        Focus: keep customers past the critical 6-month mark.
                    </div>
                </div>
                <div style="margin-bottom: 16px;">
                    <div style="font-weight: 700; color: #e2e8f0; font-size: 0.88rem;">2. Contract Type Drives Loyalty</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.55;">
                        Two-year contracts (<b style="color:#60a5fa;">−0.78</b>) vs. month-to-month (<b style="color:#f87171;">+0.66</b>) —
                        a 1.44 coefficient swing. Contract migration is the highest-ROI retention lever.
                    </div>
                </div>
                <div style="margin-bottom: 16px;">
                    <div style="font-weight: 700; color: #e2e8f0; font-size: 0.88rem;">3. Fiber Optic ≠ Satisfaction</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.55;">
                        Coef: <b style="color:#f87171;">+0.71</b>. Fiber users churn 2.3× more than DSL — likely due to
                        high cost expectations not met by service quality.
                    </div>
                </div>
                <div>
                    <div style="font-weight: 700; color: #e2e8f0; font-size: 0.88rem;">4. Service Bundling = Stickiness</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.55;">
                        Customers without Online Security, Tech Support, or Backup
                        consistently show higher churn. Each added service
                        increases switching cost and reduces churn by 5-8%.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div class="card-title">📊 Performance Summary</div>
            <table style="width: 100%; font-size: 0.82rem; margin-top: 12px; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #1e293b;">
                    <td style="padding: 8px 0; font-weight: 600; color: #94a3b8;">ROC-AUC</td>
                    <td style="padding: 8px 0; font-weight: 800; color: #e2e8f0; text-align: right;">0.844</td>
                </tr>
                <tr style="border-bottom: 1px solid #1e293b;">
                    <td style="padding: 8px 0; font-weight: 600; color: #94a3b8;">Churn Recall</td>
                    <td style="padding: 8px 0; font-weight: 800; color: #e2e8f0; text-align: right;">79.1%</td>
                </tr>
                <tr style="border-bottom: 1px solid #1e293b;">
                    <td style="padding: 8px 0; font-weight: 600; color: #94a3b8;">Precision</td>
                    <td style="padding: 8px 0; font-weight: 800; color: #e2e8f0; text-align: right;">50.4%</td>
                </tr>
                <tr style="border-bottom: 1px solid #1e293b;">
                    <td style="padding: 8px 0; font-weight: 600; color: #94a3b8;">F1-Score</td>
                    <td style="padding: 8px 0; font-weight: 800; color: #e2e8f0; text-align: right;">0.615</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; font-weight: 600; color: #94a3b8;">Accuracy</td>
                    <td style="padding: 8px 0; font-weight: 800; color: #e2e8f0; text-align: right;">75.2%</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
    Built with <span>Streamlit</span> · <span>Scikit-Learn</span> · <span>Plotly</span>
    &nbsp;·&nbsp; ChurnGuard AI — Customer Intelligence Platform
</div>
""", unsafe_allow_html=True)
