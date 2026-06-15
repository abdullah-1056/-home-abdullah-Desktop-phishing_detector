"""
PhishGuard AI — Streamlit Frontend
Dark-mode cybersecurity dashboard with:
  • Animated risk gauge (Plotly)
  • Real-time scanning with progress states
  • Suspicious token highlighting
  • URL decomposition visualization
  • Per-module confidence bars
  • Detailed explainability reports
  • Analysis history
"""
import sys
import os
import time
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Theme ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Import Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* ── Global Reset ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
  }

  /* ── Main Background ── */
  .stApp {
    background: #070b14;
    background-image:
      radial-gradient(ellipse at 20% 10%, rgba(0, 212, 255, 0.04) 0%, transparent 60%),
      radial-gradient(ellipse at 80% 80%, rgba(255, 51, 102, 0.04) 0%, transparent 60%),
      linear-gradient(180deg, #070b14 0%, #0a0f1e 100%);
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid rgba(0, 212, 255, 0.10);
  }

  /* ── Input Fields ── */
  .stTextArea textarea {
    background: #0f1628 !important;
    border: 1px solid rgba(0, 212, 255, 0.20) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
  }
  .stTextArea textarea:focus {
    border-color: rgba(0, 212, 255, 0.55) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.12) !important;
  }
  .stTextInput input {
    background: #0f1628 !important;
    border: 1px solid rgba(0, 212, 255, 0.20) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
  }
  .stTextInput input:focus {
    border-color: rgba(0, 212, 255, 0.55) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #00d4ff 0%, #0088cc 100%);
    color: #070b14;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 0.6rem 2rem;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
    font-family: 'Space Mono', monospace;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0, 212, 255, 0.35);
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(0, 212, 255, 0.15);
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 6px 6px 0 0;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(0, 212, 255, 0.08) !important;
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
  }

  /* ── Cards ── */
  .pg-card {
    background: rgba(13, 18, 32, 0.8);
    border: 1px solid rgba(0, 212, 255, 0.10);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 0.5rem 0;
    backdrop-filter: blur(8px);
  }
  .pg-card-danger {
    background: rgba(255, 30, 60, 0.06);
    border-color: rgba(255, 30, 60, 0.25);
  }
  .pg-card-safe {
    background: rgba(0, 255, 136, 0.04);
    border-color: rgba(0, 255, 136, 0.20);
  }
  .pg-card-warn {
    background: rgba(255, 180, 0, 0.05);
    border-color: rgba(255, 180, 0, 0.25);
  }

  /* ── Risk Badge ── */
  .risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .risk-high    { background: rgba(255,30,60,0.15); color: #ff3366; border: 1px solid #ff3366; }
  .risk-suspicious { background: rgba(255,140,0,0.15); color: #ff8c00; border: 1px solid #ff8c00; }
  .risk-low     { background: rgba(255,200,0,0.15); color: #ffc800; border: 1px solid #ffc800; }
  .risk-safe    { background: rgba(0,255,136,0.10); color: #00ff88; border: 1px solid #00ff88; }

  /* ── Highlight ── */
  mark.phish-highlight {
    background: rgba(255, 51, 102, 0.25);
    color: #ff8fa3;
    border-radius: 3px;
    padding: 1px 2px;
    font-weight: 600;
  }

  /* ── Monospace metric ── */
  .mono-metric {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.5px;
  }

  /* ── Score bar ── */
  .score-bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
    margin: 4px 0 2px;
  }
  .score-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease;
  }

  /* ── Section headers ── */
  .pg-section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #00d4ff;
    margin: 1.2rem 0 0.6rem;
    border-left: 3px solid #00d4ff;
    padding-left: 10px;
  }

  /* ── URL decomp table ── */
  .url-part {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 10px;
    border-radius: 6px;
    margin: 3px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
  }
  .url-part.risky { background: rgba(255,30,60,0.1); border-left: 3px solid #ff3366; }
  .url-part.safe  { background: rgba(0,255,136,0.06); border-left: 3px solid #00ff88; }
  .url-label { color: #64748b; min-width: 80px; font-size: 0.7rem; letter-spacing: 1px; text-transform: uppercase; }
  .url-value { color: #e2e8f0; flex: 1; }
  .url-icon  { font-size: 0.85rem; }

  /* ── Feature importance bars ── */
  .feat-row { display: flex; align-items: center; gap: 10px; margin: 4px 0; }
  .feat-name { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #94a3b8; min-width: 130px; }
  .feat-bar-bg { flex: 1; background: rgba(255,255,255,0.05); border-radius: 4px; height: 8px; overflow: hidden; }
  .feat-bar    { height: 100%; border-radius: 4px; }
  .feat-val  { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #64748b; min-width: 40px; text-align: right; }

  /* ── History table ── */
  .history-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; border-radius: 8px; margin: 4px 0;
    background: rgba(13,18,32,0.6); border: 1px solid rgba(255,255,255,0.05);
    font-size: 0.85rem;
  }

  /* ── Divider ── */
  hr { border-color: rgba(0, 212, 255, 0.08) !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(0, 212, 255, 0.20); border-radius: 4px; }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
if "history"        not in st.session_state: st.session_state.history = []
if "analysis_done"  not in st.session_state: st.session_state.analysis_done = False
if "last_report"    not in st.session_state: st.session_state.last_report = None
if "models_ready"   not in st.session_state: st.session_state.models_ready = False
if "detector"       not in st.session_state: st.session_state.detector = None


# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    """Load all models and return detector. Cached across sessions."""
    from core.ensemble import EnsembleDetector
    from core.transformer_engine import preload_models
    from ml.classifier import preload_classifier

    detector = EnsembleDetector()
    preload_classifier()
    preload_models()
    return detector


# ── Helper: Colour Palette ────────────────────────────────────────────────────
def score_color(score: float) -> str:
    if score >= 0.75: return "#ff3366"
    if score >= 0.45: return "#ff8c00"
    if score >= 0.20: return "#ffc800"
    return "#00ff88"

def score_class(label: str) -> str:
    m = {"High Risk": "risk-high", "Suspicious": "risk-suspicious",
         "Low Risk": "risk-low", "Safe": "risk-safe"}
    return m.get(label, "risk-safe")


# ── Plotly Gauge ──────────────────────────────────────────────────────────────
def make_gauge(probability: float, label: str) -> go.Figure:
    color = score_color(probability)
    pct   = probability * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={
            "suffix": "%",
            "font": {"size": 44, "color": color, "family": "Space Mono"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "#1e293b",
                "tickwidth": 1,
                "tickfont": {"color": "#475569", "size": 10, "family": "DM Sans"},
            },
            "bar":             {"color": color, "thickness": 0.28},
            "bgcolor":         "rgba(0,0,0,0)",
            "borderwidth":     0,
            "steps": [
                {"range": [0,  20],  "color": "rgba(0,255,136,0.07)"},
                {"range": [20, 45],  "color": "rgba(255,200,0,0.07)"},
                {"range": [45, 75],  "color": "rgba(255,140,0,0.08)"},
                {"range": [75, 100], "color": "rgba(255,30,60,0.10)"},
            ],
            "threshold": {
                "line":  {"color": color, "width": 3},
                "thickness": 0.88,
                "value": pct,
            },
        },
        title={
            "text": f"<b>PHISHING PROBABILITY</b><br>"
                    f"<span style='font-size:13px;color:#64748b;font-family:DM Sans'>{label}</span>",
            "font": {"size": 13, "color": "#94a3b8", "family": "Space Mono"},
        },
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=10, l=20, r=20),
        height=260,
    )
    return fig


# ── Plotly Radar ──────────────────────────────────────────────────────────────
def make_radar(scores: dict) -> go.Figure:
    cats  = list(scores.keys())
    vals  = [scores[c] * 100 for c in cats]
    vals += [vals[0]]   # Close polygon
    cats += [cats[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats,
        fill="toself",
        fillcolor="rgba(255, 51, 102, 0.12)",
        line={"color": "#ff3366", "width": 2},
        marker={"color": "#ff3366", "size": 6},
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont={"size": 8, "color": "#475569"},
                gridcolor="rgba(255,255,255,0.06)",
                linecolor="rgba(255,255,255,0.06)",
            ),
            angularaxis=dict(
                tickfont={"size": 10, "color": "#94a3b8", "family": "DM Sans"},
                gridcolor="rgba(255,255,255,0.06)",
                linecolor="rgba(255,255,255,0.06)",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20, l=40, r=40),
        height=240,
        showlegend=False,
    )
    return fig


# ── Score Bar HTML ─────────────────────────────────────────────────────────────
def score_bar_html(label: str, score: float, icon: str = "") -> str:
    color = score_color(score)
    pct   = int(score * 100)
    return f"""
    <div style="margin: 6px 0;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:3px;">
        <span style="font-size:0.82rem; color:#94a3b8;">{icon} {label}</span>
        <span style="font-family:'Space Mono',monospace; font-size:0.78rem; color:{color};">{pct}%</span>
      </div>
      <div class="score-bar-wrap">
        <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
      </div>
    </div>"""


# ── URL Decomposition HTML ────────────────────────────────────────────────────
def url_decomp_html(decomp: dict) -> str:
    rows = []
    icons = {
        "scheme": "🔒", "subdomain": "📂", "domain": "🌐",
        "tld": "🏷", "path": "📁", "query": "🔍",
        "entropy": "📊", "length": "📏",
    }
    for key, info in decomp.items():
        risky  = info.get("risk", False)
        cls    = "risky" if risky else "safe"
        icon   = "🔴" if risky else "🟢"
        badge  = icons.get(key, "")
        rows.append(f"""
        <div class="url-part {cls}">
          <span class="url-icon">{icon}</span>
          <span class="url-label">{badge} {key}</span>
          <span class="url-value">{info['value']}</span>
        </div>""")
    return "".join(rows)


# ── Risk Factor List HTML ──────────────────────────────────────────────────────
def factor_list_html(factors: list[str], is_risk: bool) -> str:
    if not factors:
        msg = "No risk factors detected." if is_risk else "No safe signals detected."
        return f'<div style="color:#475569;font-size:0.85rem;padding:4px 0;">{msg}</div>'

    color  = "#ff6b8a" if is_risk else "#4ade80"
    bg     = "rgba(255,51,102,0.05)" if is_risk else "rgba(0,255,136,0.04)"
    border = "rgba(255,51,102,0.15)" if is_risk else "rgba(0,255,136,0.12)"

    rows = []
    for f in factors:
        rows.append(
            f'<div style="padding:7px 12px; margin:3px 0; border-radius:6px; '
            f'background:{bg}; border:1px solid {border}; '
            f'font-size:0.83rem; color:{color}; line-height:1.4;">{f}</div>'
        )
    return "".join(rows)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem;">
      <div style="font-family:'Space Mono',monospace; font-size:1.2rem; font-weight:700;
                  color:#00d4ff; letter-spacing:1px;">🛡 PhishGuard</div>
      <div style="font-size:0.72rem; color:#475569; letter-spacing:2px; text-transform:uppercase;
                  margin-top:2px;">AI Phishing Detection</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="pg-section-title">Navigation</div>', unsafe_allow_html=True)
    mode = st.radio(
        "",
        ["Email Analysis", "URL Scanner", "Analysis History"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="pg-section-title">Model Status</div>', unsafe_allow_html=True)

    with st.spinner("Loading AI models…"):
        detector = load_system()

    st.markdown("""
    <div style="display:flex;flex-direction:column;gap:5px;margin-top:4px;">
      <div style="display:flex;align-items:center;gap:8px;font-size:0.78rem;color:#94a3b8;">
        <span style="color:#00ff88;">●</span> all-MiniLM-L6-v2
      </div>
      <div style="display:flex;align-items:center;gap:8px;font-size:0.78rem;color:#94a3b8;">
        <span style="color:#00ff88;">●</span> NLI Zero-Shot
      </div>
      <div style="display:flex;align-items:center;gap:8px;font-size:0.78rem;color:#94a3b8;">
        <span style="color:#00ff88;">●</span> RandomForest + XGBoost
      </div>
      <div style="display:flex;align-items:center;gap:8px;font-size:0.78rem;color:#94a3b8;">
        <span style="color:#00ff88;">●</span> URL Heuristics Engine
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="pg-section-title">Ensemble Weights</div>', unsafe_allow_html=True)
    from config.settings import (
        WEIGHT_TRANSFORMER, WEIGHT_HEURISTICS, WEIGHT_URL, WEIGHT_CLASSIFIER,
    )
    st.markdown(
        score_bar_html("Transformer", WEIGHT_TRANSFORMER, "🤖") +
        score_bar_html("Heuristics",  WEIGHT_HEURISTICS,  "📋") +
        score_bar_html("URL Analysis", WEIGHT_URL,         "🔗") +
        score_bar_html("ML Classifier", WEIGHT_CLASSIFIER, "🌳"),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:#334155;text-align:center;line-height:1.6;">
      PhishGuard AI v1.0<br>
      CPU-optimized · Offline capable<br>
      No data sent externally
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: Email Analysis
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Email Analysis":

    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <h1 style="font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;
                 color:#e2e8f0;margin:0;letter-spacing:-0.5px;">
        📧 Email Threat Analysis
      </h1>
      <p style="color:#64748b;font-size:0.88rem;margin:4px 0 0;">
        Paste a suspicious email below. Supports plain text, RFC-822 format, and EML content.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sample emails ─────────────────────────────────────────────────────────
    SAMPLES = {
        "🎣 Classic PayPal Phish": """From: "PayPal Security" <security@paypa1-secure.tk>
To: customer@gmail.com
Subject: URGENT: Your PayPal account has been limited!
Reply-To: paypal.recovery@gmail-helpdesk.xyz

Dear Valued Customer,

We have detected unusual activity on your PayPal account. Your account has been temporarily LIMITED due to unauthorized access attempts.

ACTION REQUIRED: You must verify your account within 24 hours or your account will be permanently suspended.

Click here to verify your account immediately: http://paypa1-secure.tk/verify?redirect=account&token=829182

Please provide the following information:
- Full name and date of birth
- Credit card number and CVV
- Social security number
- Current password

Failure to comply will result in permanent account termination.

PayPal Security Team""",
        "💼 Legitimate Business Email": """From: Sarah Mitchell <s.mitchell@acmecorp.com>
To: team@company.com
Subject: Q4 Project Update - Action Items

Hi everyone,

Just a quick update on the Q4 roadmap. We had a productive session yesterday and I wanted to share the key takeaways before our next sync.

The main deliverables for this quarter remain on track. The engineering team has completed the initial sprint and we are moving into QA this week. Marketing has finalized the campaign assets and is ready for the launch timeline.

Action items:
- Review the attached specification document by Friday
- Confirm availability for the stakeholder demo on the 15th
- Submit your department budget forecasts by end of week

Please reach out if you have any questions. Looking forward to a strong finish to the quarter.

Best regards,
Sarah Mitchell
Director of Operations, Acme Corp""",
        "🏦 Bank Credential Phish": """From: Chase Bank <noreply@chase-secure-alert.ml>
Subject: ⚠️ SECURITY ALERT: Suspicious Login Detected on Your Account

Dear Chase Customer,

We detected a suspicious login attempt from an unrecognized device in Lagos, Nigeria at 3:42 AM.

To protect your account, we have temporarily suspended access. You must confirm your identity immediately to restore access and avoid permanent suspension.

VERIFY NOW: http://192.168.1.254/chase/verify.php?session=a7f2b

Enter your:
✓ Online Banking Username & Password
✓ Debit Card Number + PIN
✓ Social Security Number
✓ Mother's Maiden Name

This link expires in 1 HOUR. Act immediately!

Chase Online Security Team""",
    }

    col_inp, col_sample = st.columns([3, 1])
    with col_sample:
        sample_choice = st.selectbox(
            "Load sample",
            ["(paste your own)"] + list(SAMPLES.keys()),
        )

    default_text = SAMPLES.get(sample_choice, "") if sample_choice != "(paste your own)" else ""

    email_input = st.text_area(
        "Email Content",
        value=default_text,
        height=260,
        placeholder="Paste the full email here (including headers if available)…",
        label_visibility="collapsed",
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("🔍 Scan Email", use_container_width=True)
    with col_info:
        st.markdown(
            '<div style="padding:0.6rem 0;font-size:0.82rem;color:#475569;">'
            'Analysis runs locally · No email content is transmitted · ~2-8 seconds first run</div>',
            unsafe_allow_html=True,
        )

    if run_btn and email_input.strip():
        # ── Scanning Animation ─────────────────────────────────────────────
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            prog = st.progress(0)
            status = st.empty()
            steps = [
                (15, "🔍 Parsing email structure and headers…"),
                (35, "📋 Running heuristic rule engine (40+ rules)…"),
                (55, "🤖 Running transformer semantic analysis…"),
                (75, "🔗 Analyzing embedded URLs…"),
                (90, "🌳 Ensemble ML classifier scoring…"),
                (100, "✅ Generating explainability report…"),
            ]
            for pct, msg in steps:
                status.markdown(
                    f'<div style="font-size:0.85rem;color:#64748b;">{msg}</div>',
                    unsafe_allow_html=True,
                )
                prog.progress(pct / 100)
                time.sleep(0.3)

        progress_placeholder.empty()

        # ── Run actual analysis ───────────────────────────────────────────
        with st.spinner(""):
            report = detector.analyze_email(email_input)

        # Save to history
        st.session_state.history.insert(0, {
            "type": "📧 Email",
            "label": report.risk_label,
            "score": report.phishing_probability,
            "time_ms": report.analysis_time_ms,
            "preview": email_input[:60] + "…",
        })

        # ── Results Layout ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="pg-section-title">Verdict</div>', unsafe_allow_html=True)

        col_gauge, col_verdict = st.columns([1, 1.6])

        with col_gauge:
            st.plotly_chart(
                make_gauge(report.phishing_probability, report.risk_label),
                use_container_width=True, config={"displayModeBar": False},
            )

        with col_verdict:
            badge_class = score_class(report.risk_label)
            color = score_color(report.phishing_probability)
            st.markdown(f"""
            <div class="pg-card" style="height:100%;">
              <div class="risk-badge {badge_class}" style="margin-bottom:14px;">{report.risk_label}</div>
              <div style="margin-bottom:10px;">
                <span class="mono-metric" style="color:{color};">
                  {report.phishing_probability:.1%}
                </span>
                <span style="color:#475569;font-size:0.82rem;margin-left:8px;">
                  phishing probability
                </span>
              </div>
              <div style="font-size:0.82rem;color:#64748b;margin-bottom:14px;">
                Confidence: <span style="color:#94a3b8;font-weight:600;">{report.confidence:.0%}</span>
                &nbsp;·&nbsp; Analyzed in: <span style="color:#94a3b8;font-weight:600;">{report.analysis_time_ms:.0f} ms</span>
              </div>
              <hr style="margin:10px 0;">
              {score_bar_html("AI Transformer", report.transformer_score, "🤖")}
              {score_bar_html("Heuristic Rules", report.heuristic_score, "📋")}
              {score_bar_html("URL Risk", report.url_score, "🔗")}
              {score_bar_html("ML Classifier", report.classifier_score, "🌳")}
            </div>
            """, unsafe_allow_html=True)

        # ── Detail Tabs ───────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "⚠ Risk Factors", "✓ Safe Signals", "🔗 URL Analysis", "📊 Model Details",
        ])

        with tab1:
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown('<div class="pg-section-title">Risk Factors Detected</div>', unsafe_allow_html=True)
                st.markdown(factor_list_html(report.all_risk_factors, is_risk=True), unsafe_allow_html=True)
            with col_r2:
                st.markdown('<div class="pg-section-title">Highlighted Email Body</div>', unsafe_allow_html=True)
                highlighted = report.highlighted_body or report.parsed_email.body_plain if report.parsed_email else ""
                highlighted_html = highlighted[:3000].replace("\n", "<br>")
                st.markdown(
                    f'<div class="pg-card" style="font-size:0.82rem;line-height:1.65;'
                    f'max-height:320px;overflow-y:auto;font-family:\'DM Sans\',sans-serif;">'
                    f'{highlighted_html}</div>',
                    unsafe_allow_html=True,
                )

        with tab2:
            st.markdown('<div class="pg-section-title">Safe Signals</div>', unsafe_allow_html=True)
            st.markdown(factor_list_html(report.all_safe_factors, is_risk=False), unsafe_allow_html=True)

            # Radar chart
            if report.heuristic_result:
                hr = report.heuristic_result
                radar_scores = {
                    "Urgency":    hr.urgency_score,
                    "Spoofing":   hr.spoofing_score,
                    "Content":    hr.content_score,
                    "Structural": hr.structural_score,
                    "Links":      report.url_score,
                    "Transformer": report.transformer_score,
                }
                st.markdown('<div class="pg-section-title">Threat Radar</div>', unsafe_allow_html=True)
                st.plotly_chart(
                    make_radar(radar_scores),
                    use_container_width=True, config={"displayModeBar": False},
                )

        with tab3:
            if report.url_results:
                for ur in report.url_results:
                    risk_color = score_color(ur.risk_score)
                    st.markdown(
                        f'<div class="pg-card" style="margin-bottom:10px;">'
                        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">'
                        f'<code style="font-size:0.78rem;color:#94a3b8;word-break:break-all;">{ur.url[:80]}</code>'
                        f'<span style="font-family:\'Space Mono\',monospace;color:{risk_color};'
                        f'font-size:0.9rem;font-weight:700;white-space:nowrap;margin-left:10px;">'
                        f'{ur.risk_score:.0%} risk</span>'
                        f'</div>'
                        f'{url_decomp_html(ur.decomposition)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="color:#475569;font-size:0.88rem;padding:1rem 0;">'
                    'No URLs detected in this email.</div>',
                    unsafe_allow_html=True,
                )

        with tab4:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown('<div class="pg-section-title">Transformer Model</div>', unsafe_allow_html=True)
                if report.transformer_result:
                    tr = report.transformer_result
                    st.markdown(f"""
                    <div class="pg-card">
                      <div style="font-size:0.8rem;color:#64748b;margin-bottom:8px;">Model</div>
                      <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#94a3b8;margin-bottom:12px;">{tr.model_used}</div>
                      {score_bar_html("Embedding Score", tr.embedding_score, "📐")}
                      {score_bar_html("Zero-Shot NLI", tr.zero_shot_score, "🧠")}
                      <div style="font-size:0.78rem;color:#475569;margin-top:10px;">
                        Inference: <span style="color:#94a3b8;">{tr.inference_time_ms:.0f} ms</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if tr.top_tokens:
                        st.markdown('<div class="pg-section-title">Key Suspicious Tokens</div>', unsafe_allow_html=True)
                        tokens_html = " ".join(
                            f'<span style="background:rgba(255,51,102,0.15);color:#ff8fa3;'
                            f'border-radius:4px;padding:3px 8px;font-size:0.78rem;'
                            f'margin:2px;display:inline-block;">{t}</span>'
                            for t in tr.top_tokens[:8]
                        )
                        st.markdown(tokens_html, unsafe_allow_html=True)

            with col_m2:
                st.markdown('<div class="pg-section-title">Ensemble Weights Used</div>', unsafe_allow_html=True)
                weight_data = pd.DataFrame({
                    "Module":    ["Transformer", "Heuristics", "URL Analysis", "ML Classifier"],
                    "Weight":    [report.weights_used.get(k, 0) for k in ["transformer", "heuristics", "url", "classifier"]],
                    "Score":     [report.transformer_score, report.heuristic_score, report.url_score, report.classifier_score],
                })
                weight_data["Contribution"] = weight_data["Weight"] * weight_data["Score"]

                fig = px.bar(
                    weight_data, x="Module", y="Contribution",
                    color="Contribution",
                    color_continuous_scale=["#00ff88", "#ffc800", "#ff8c00", "#ff3366"],
                    text=weight_data["Contribution"].apply(lambda x: f"{x:.1%}"),
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(tickfont={"color": "#64748b", "size": 10}),
                    yaxis=dict(tickfont={"color": "#64748b", "size": 10},
                               gridcolor="rgba(255,255,255,0.04)"),
                    margin=dict(t=10, b=10, l=10, r=10), height=200,
                    coloraxis_showscale=False,
                    font={"family": "DM Sans"},
                )
                fig.update_traces(textfont_size=10, textposition="outside")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    elif run_btn:
        st.warning("Please paste an email to analyze.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: URL Scanner
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "URL Scanner":

    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <h1 style="font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;
                 color:#e2e8f0;margin:0;">🔗 URL Threat Scanner</h1>
      <p style="color:#64748b;font-size:0.88rem;margin:4px 0 0;">
        Analyze URLs for structural phishing indicators, typosquatting, and entropy anomalies.
      </p>
    </div>
    """, unsafe_allow_html=True)

    url_input = st.text_input(
        "URL",
        placeholder="https://example.com or paste a suspicious link…",
        label_visibility="collapsed",
    )

    # Bulk URL scanner
    with st.expander("📋 Bulk URL Scanner (paste multiple URLs, one per line)"):
        bulk_input = st.text_area("Bulk URLs", height=120, label_visibility="collapsed")

    col_b1, col_b2 = st.columns([1, 4])
    with col_b1:
        url_btn = st.button("🔍 Scan URL", use_container_width=True)

    if url_btn:
        urls_to_scan = []
        if url_input.strip():
            urls_to_scan.append(url_input.strip())
        if bulk_input.strip():
            urls_to_scan.extend([u.strip() for u in bulk_input.strip().split("\n") if u.strip()])

        if not urls_to_scan:
            st.warning("Please enter at least one URL.")
        else:
            for url in urls_to_scan[:10]:  # Max 10 at a time
                with st.spinner(f"Scanning {url[:60]}…"):
                    report = detector.analyze_url(url)

                ur = report.url_results[0] if report.url_results else None
                risk_color = score_color(report.phishing_probability)

                st.session_state.history.insert(0, {
                    "type": "🔗 URL",
                    "label": report.risk_label,
                    "score": report.phishing_probability,
                    "time_ms": report.analysis_time_ms,
                    "preview": url[:60] + ("…" if len(url) > 60 else ""),
                })

                # Result card
                badge_cls = score_class(report.risk_label)
                st.markdown(f"""
                <div class="pg-card {'pg-card-danger' if report.phishing_probability >= 0.75
                    else 'pg-card-warn' if report.phishing_probability >= 0.45
                    else 'pg-card-safe'}" style="margin-bottom:16px;">
                  <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:12px;">
                    <div>
                      <code style="font-size:0.82rem;color:#94a3b8;word-break:break-all;">{url}</code>
                    </div>
                    <div style="text-align:right;margin-left:16px;flex-shrink:0;">
                      <div class="risk-badge {badge_cls}">{report.risk_label}</div>
                      <div style="font-family:'Space Mono',monospace;font-size:1.4rem;
                                  font-weight:700;color:{risk_color};margin-top:4px;">
                        {report.phishing_probability:.0%}
                      </div>
                    </div>
                  </div>
                """, unsafe_allow_html=True)

                if ur:
                    st.markdown(
                        f'<div style="margin-bottom:8px;">{url_decomp_html(ur.decomposition)}</div>',
                        unsafe_allow_html=True,
                    )

                col_rf, col_sf = st.columns(2)
                with col_rf:
                    st.markdown(factor_list_html(report.all_risk_factors[:8], is_risk=True), unsafe_allow_html=True)
                with col_sf:
                    st.markdown(factor_list_html(report.all_safe_factors[:6], is_risk=False), unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: History
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Analysis History":

    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <h1 style="font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;
                 color:#e2e8f0;margin:0;">📜 Analysis History</h1>
      <p style="color:#64748b;font-size:0.88rem;margin:4px 0 0;">
        All scans performed in this session.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            '<div class="pg-card" style="text-align:center;padding:2rem;color:#475569;">'
            'No analyses yet. Scan an email or URL to see results here.</div>',
            unsafe_allow_html=True,
        )
    else:
        # Summary stats
        history = st.session_state.history
        n_total   = len(history)
        n_phish   = sum(1 for h in history if h["score"] >= 0.45)
        n_safe    = n_total - n_phish
        avg_score = sum(h["score"] for h in history) / n_total

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, color in [
            (c1, "Total Scanned",   str(n_total),          "#94a3b8"),
            (c2, "Threats Found",   str(n_phish),          "#ff3366"),
            (c3, "Safe",            str(n_safe),            "#00ff88"),
            (c4, "Avg Risk Score",  f"{avg_score:.1%}",    score_color(avg_score)),
        ]:
            with col:
                st.markdown(f"""
                <div class="pg-card" style="text-align:center;padding:1rem;">
                  <div style="font-size:0.72rem;color:#475569;letter-spacing:1px;text-transform:uppercase;">{label}</div>
                  <div style="font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;color:{color};margin-top:4px;">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="pg-section-title">Scan Log</div>', unsafe_allow_html=True)

        for i, h in enumerate(history):
            color = score_color(h["score"])
            badge = score_class(h["label"])
            st.markdown(f"""
            <div class="history-row">
              <div style="display:flex;align-items:center;gap:12px;">
                <span style="font-size:0.8rem;color:#334155;font-family:'Space Mono',monospace;">#{i+1:02d}</span>
                <span style="font-size:0.8rem;color:#64748b;">{h['type']}</span>
                <span style="font-size:0.82rem;color:#94a3b8;max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{h['preview']}</span>
              </div>
              <div style="display:flex;align-items:center;gap:14px;flex-shrink:0;">
                <span style="font-family:'Space Mono',monospace;font-size:0.82rem;color:{color};">{h['score']:.0%}</span>
                <span class="risk-badge {badge}" style="font-size:0.68rem;">{h['label']}</span>
                <span style="font-size:0.75rem;color:#334155;">{h['time_ms']:.0f}ms</span>
              </div>
            </div>""", unsafe_allow_html=True)

        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()

        # Distribution chart
        if n_total >= 2:
            st.markdown('<div class="pg-section-title">Risk Distribution</div>', unsafe_allow_html=True)
            scores = [h["score"] * 100 for h in history]
            fig = go.Figure(go.Histogram(
                x=scores, nbinsx=10,
                marker_color=[score_color(s/100) for s in scores],
                opacity=0.8,
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Risk Score (%)", tickfont={"color": "#64748b"},
                           gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Count", tickfont={"color": "#64748b"},
                           gridcolor="rgba(255,255,255,0.04)"),
                margin=dict(t=10, b=30, l=30, r=10), height=200,
                font={"family": "DM Sans", "color": "#64748b"},
                bargap=0.1,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING STATUS shown in sidebar (injected at bottom of sidebar block)
# ══════════════════════════════════════════════════════════════════════════════
# NOTE: This block runs after the sidebar definition above.
# We re-open the sidebar context to append the training status card.
import json
from pathlib import Path as _Path

_summary_path = _Path(__file__).parent.parent / "models" / "training_summary.json"
_report_path  = _Path(__file__).parent.parent / "reports" / "training_report.png"

with st.sidebar:
    st.markdown('<div class="pg-section-title">Training Status</div>', unsafe_allow_html=True)

    if _summary_path.exists():
        with open(_summary_path) as _f:
            _s = json.load(_f)
        _bert = _s.get("distilbert", {})
        _rf   = _s.get("random_forest", {})
        _xgb  = _s.get("xgboost", {})
        _ens  = _s.get("url_ensemble", {})

        st.markdown(f"""
        <div style="background:rgba(0,255,136,0.05);border:1px solid rgba(0,255,136,0.20);
                    border-radius:8px;padding:10px 12px;font-size:0.76rem;">
          <div style="color:#00ff88;font-weight:700;margin-bottom:6px;">✅ Trained on Real Data</div>
          <div style="color:#64748b;margin-bottom:4px;">Samples: {_s.get('email_samples',0):,} emails · {_s.get('url_samples',0):,} URLs</div>
          {"<div style='color:#94a3b8;'>🤖 DistilBERT F1=<b style=color:#00d4ff>" + str(round(_bert.get('f1',0),4)) + "</b> AUC=<b style=color:#00d4ff>" + str(round(_bert.get('roc_auc',0),4)) + "</b></div>" if not _bert.get('skipped') else "<div style='color:#475569;'>🤖 DistilBERT — skipped</div>"}
          <div style="color:#94a3b8;">🌳 RF F1=<b style="color:#00d4ff">{round(_rf.get('f1',0),4)}</b> AUC=<b style="color:#00d4ff">{round(_rf.get('roc_auc',0),4)}</b></div>
          <div style="color:#94a3b8;">⚡ XGB F1=<b style="color:#00d4ff">{round(_xgb.get('f1',0),4)}</b> AUC=<b style="color:#00d4ff">{round(_xgb.get('roc_auc',0),4)}</b></div>
          <div style="color:#475569;margin-top:4px;font-size:0.7rem;">{_s.get('trained_at','')}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(255,180,0,0.05);border:1px solid rgba(255,180,0,0.20);
                    border-radius:8px;padding:10px 12px;font-size:0.76rem;">
          <div style="color:#ffc800;font-weight:700;margin-bottom:4px;">⚠ Using Synthetic Data</div>
          <div style="color:#64748b;line-height:1.5;">
            Run training to use your real dataset:<br>
            <code style="color:#00d4ff;font-size:0.72rem;">python3 train.py --dataset MASTER_phishing_dataset.csv</code>
          </div>
        </div>
        """, unsafe_allow_html=True)

# Inject Training Report page into the mode selector
# (The mode radio was already rendered — we handle it via a new condition below)
if mode == "Email Analysis" or mode == "URL Scanner" or mode == "Analysis History":
    pass  # handled above

# Training Report page
if _report_path.exists() and "📊 Training Report" in (mode or ""):
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <h1 style="font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;color:#e2e8f0;margin:0;">
        📊 Training Report
      </h1>
    </div>
    """, unsafe_allow_html=True)
    st.image(str(_report_path), use_container_width=True)
