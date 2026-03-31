import streamlit as st

# ── MUST be first ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LMS Intelligence · OULAD",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.dashboard  import show_dashboard
from components.patterns   import show_patterns
from components.prediction import show_prediction
from components.model_results import show_model_results

def _inject_global_css():
    st.markdown(
        """<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">"""
        """<style>"""
        """.stApp{background:#0B0F19!important;font-family:'Inter',sans-serif!important}"""
        """html,body,[class*="css"]{font-family:'Inter',sans-serif}"""
        """[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1321 0%,#0B0F19 100%)!important;border-right:1px solid rgba(56,189,248,.12)!important;padding-top:0!important}"""
        """[data-testid="stSidebar"]>div:first-child{padding-top:0!important}"""
        """#MainMenu{visibility:hidden}footer{visibility:hidden}header{visibility:hidden}"""
        """[data-testid="stVerticalBlock"]{gap:0rem}"""
        """div.block-container{padding-top:1.5rem!important}"""
        """div[data-testid="metric-container"]{background:linear-gradient(135deg,rgba(17,24,39,.9) 0%,rgba(15,18,30,.95) 100%)!important;border:1px solid rgba(56,189,248,.18)!important;border-radius:16px!important;padding:20px!important;transition:all .25s ease;box-shadow:0 4px 24px rgba(0,0,0,.35)}"""
        """div[data-testid="metric-container"]:hover{border-color:rgba(56,189,248,.5)!important;transform:translateY(-3px);box-shadow:0 8px 32px rgba(56,189,248,.12)}"""
        """div[data-testid="metric-container"] [data-testid="stMetricLabel"]{font-size:.62rem!important;font-weight:700!important;color:#4B5563!important;text-transform:uppercase;letter-spacing:2px}"""
        """div[data-testid="metric-container"] [data-testid="stMetricValue"]{font-size:2rem!important;font-weight:800!important;color:#F1F5F9!important;font-family:'JetBrains Mono',monospace!important}"""
        """[data-testid="stDataFrame"] iframe{border-radius:12px}"""
        """.stButton>button{background:linear-gradient(135deg,#0EA5E9,#38BDF8)!important;color:#0B0F19!important;border:none!important;border-radius:12px!important;font-weight:700!important;letter-spacing:.4px!important;padding:.65rem 2rem!important;transition:all .2s ease!important;box-shadow:0 4px 20px rgba(56,189,248,.3)!important}"""
        """.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 30px rgba(56,189,248,.45)!important}"""
        """[data-testid="stNumberInput"] input{background:#111827!important;border:1px solid rgba(56,189,248,.2)!important;border-radius:10px!important;color:#F1F5F9!important}"""
        """[data-testid="stExpander"]{background:rgba(17,24,39,.6)!important;border:1px solid rgba(56,189,248,.1)!important;border-radius:14px!important}"""
        """</style>""",
        unsafe_allow_html=True,
    )

_inject_global_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / Brand
    st.markdown("""
    <div style="padding:28px 16px 20px 16px; border-bottom:1px solid rgba(56,189,248,0.1); margin-bottom:16px;">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:42px;height:42px;border-radius:12px;
                        background:linear-gradient(135deg,#0EA5E9,#6366F1);
                        display:flex;align-items:center;justify-content:center;
                        font-size:1.3rem;box-shadow:0 4px 16px rgba(14,165,233,0.35);">
                🎓
            </div>
            <div>
                <div style="font-size:1.05rem;font-weight:800;color:#F1F5F9;line-height:1.2;">LMS Intelligence</div>
                <div style="font-size:0.65rem;color:#4B5563;letter-spacing:1.5px;text-transform:uppercase;margin-top:2px;">OULAD · Analytics</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.6rem;color:#374151;text-transform:uppercase;letter-spacing:2px;margin-left:4px;margin-bottom:6px;">Navigation</p>', unsafe_allow_html=True)

    page = st.radio(
        "page",
        ["📊  Dashboard", "🔍  Pattern Engine", "📈  Model Results", "🤖  Live Prediction"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style="position:fixed;bottom:20px;left:0;width:240px;padding:0 16px;">
        <div style="background:rgba(17,24,39,0.6);border:1px solid rgba(56,189,248,0.1);
                    border-radius:12px;padding:12px 14px;">
            <div style="font-size:0.62rem;color:#374151;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">Model Status</div>
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:8px;height:8px;border-radius:50%;background:#4ADE80;
                            box-shadow:0 0 8px #4ADE80;animation:pulse 2s infinite;"></div>
                <span style="font-size:0.8rem;color:#D1D5DB;font-weight:600;">RandomForest · Hybrid · F1 0.8222</span>
            </div>
        </div>
    </div>
    <style>
    @keyframes pulse {0%,100%{opacity:1}50%{opacity:0.4}}
    </style>
    """, unsafe_allow_html=True)

# ── Route ─────────────────────────────────────────────────────────────────────
if "Dashboard" in page:
    show_dashboard()
elif "Pattern" in page:
    show_patterns()
elif "Model Results" in page:
    show_model_results()
elif "Prediction" in page:
    show_prediction()