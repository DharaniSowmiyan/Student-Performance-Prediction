import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

ACCENT = "#38BDF8"
GREEN  = "#34D399"
RED    = "#F87171"
PURPLE = "#A78BFA"


def _section(icon, title, subtitle=""):
    sub_html = f'<div style="font-size:0.72rem;color:#4B5563;">{subtitle}</div>' if subtitle else ""
    html = (
        '<div style="display:flex;align-items:center;gap:12px;margin:28px 0 18px 0;'
        'padding-bottom:12px;border-bottom:1px solid rgba(56,189,248,0.1);">'
        '<div style="width:36px;height:36px;border-radius:10px;'
        'background:linear-gradient(135deg,#0EA5E9,#6366F1);'
        'display:flex;align-items:center;justify-content:center;font-size:0.95rem;'
        f'box-shadow:0 3px 12px rgba(14,165,233,0.3);">{icon}</div>'
        f'<div><div style="font-size:1rem;font-weight:700;color:#F1F5F9;">{title}</div>'
        f'{sub_html}</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pkl_path = os.path.join(base, "results", "best_model.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def show_prediction():
    st.markdown("""
    <div style="padding:8px 0 24px 0;">
        <div style="font-size:0.65rem;color:#4B5563;text-transform:uppercase;letter-spacing:2.5px;margin-bottom:8px;">Real-Time Analysis</div>
        <h1 style="font-size:2.2rem;font-weight:800;margin:0;
                   background:linear-gradient(135deg,#F1F5F9 0%,#38BDF8 40%,#A78BFA 100%);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Live Prediction
        </h1>
        <p style="color:#4B5563;font-size:0.85rem;margin-top:6px;">
            Input a student's LMS behaviour profile to predict their performance outcome using the trained Random Forest model
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        model_data = load_model()
        pipeline   = model_data["pipeline"]
        cols       = model_data["feature_cols"]
        model_name = model_data["model_name"]
        feat_set   = model_data["feature_set"]
        metrics    = model_data["metrics"]

        # Separate static and pattern columns
        static_cols  = [c for c in cols if not c.startswith("pat_")]
        pattern_cols = [c for c in cols if c.startswith("pat_")]

        # ── Model badge ────────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:rgba(17,24,39,0.7);border:1px solid rgba(56,189,248,0.15);
                    border-radius:14px;padding:16px 20px;margin-bottom:24px;
                    display:flex;align-items:center;gap:16px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#4ADE80;
                        box-shadow:0 0 8px #4ADE80;flex-shrink:0;"></div>
            <div>
                <span style="font-size:0.7rem;color:#4B5563;text-transform:uppercase;letter-spacing:1.5px;">Active Model · </span>
                <span style="font-size:0.85rem;color:#38BDF8;font-weight:700;">{model_name}</span>
                <span style="font-size:0.7rem;color:#4B5563;"> · Feature Set: </span>
                <span style="font-size:0.85rem;color:#A78BFA;font-weight:700;">{feat_set.upper()}</span>
            </div>
            <div style="margin-left:auto;display:flex;gap:14px;">
                <div style="text-align:center;">
                    <div style="font-size:0.6rem;color:#4B5563;text-transform:uppercase;letter-spacing:1px;">F1</div>
                    <div style="font-size:1rem;font-weight:700;color:#F1F5F9;font-family:'JetBrains Mono',monospace;">{metrics.get('f1',0):.4f}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:0.6rem;color:#4B5563;text-transform:uppercase;letter-spacing:1px;">Accuracy</div>
                    <div style="font-size:1rem;font-weight:700;color:#F1F5F9;font-family:'JetBrains Mono',monospace;">{metrics.get('accuracy',0):.4f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("predict_form"):
            col_left, col_right = st.columns(2)

            with col_left:
                _section("📊", "Static Interaction Features", "Numeric activity counts per student")
                feature_values = {}
                for c in static_cols:
                    label = c.replace("_interactions", " Interactions").replace("_", " ").title()
                    if "clicks" in c.lower():
                        feature_values[c] = st.number_input(label, min_value=0, max_value=50000, value=500, step=10, key=c)
                    else:
                        feature_values[c] = st.number_input(label, min_value=0, max_value=10000, value=50, step=5, key=c)

            with col_right:
                if pattern_cols:
                    _section("🔀", "Sequential Pattern Flags", "Did the student exhibit these sequences?")
                    for c in pattern_cols:
                        raw_label = c.replace("pat_", "").replace("_", " → ")
                        feature_values[c] = int(st.checkbox(f"✦ {raw_label}", key=c))

            st.markdown("<br>", unsafe_allow_html=True)

            # Centered submit button
            sc1, sc2, sc3 = st.columns([1.5, 1, 1.5])
            with sc2:
                submitted = st.form_submit_button(
                    "⚡  Analyze Student",
                    use_container_width=True,
                )

        # ── Result ─────────────────────────────────────────────────────────────
        if submitted:
            input_df = pd.DataFrame([{c: feature_values[c] for c in cols}])
            pred     = pipeline.predict(input_df)[0]
            proba    = pipeline.predict_proba(input_df)[0]
            high_p   = proba[1] * 100
            low_p    = proba[0] * 100

            if pred == 1:
                st.markdown(f"""
                <div style="margin-top:24px;padding:32px;border-radius:20px;
                            background:linear-gradient(135deg,rgba(52,211,153,0.08),rgba(56,189,248,0.05));
                            border:2px solid rgba(52,211,153,0.45);text-align:center;
                            box-shadow:0 0 40px rgba(52,211,153,0.12);">
                    <div style="font-size:3rem;margin-bottom:8px;">🎓</div>
                    <div style="font-size:2rem;font-weight:800;color:#34D399;letter-spacing:-0.5px;">HIGH PERFORMER</div>
                    <div style="font-size:0.85rem;color:#4B5563;margin-top:6px;">
                        The model predicts this student is on-track for strong academic performance
                    </div>
                    <div style="margin-top:20px;display:flex;justify-content:center;gap:32px;">
                        <div>
                            <div style="font-size:0.6rem;color:#4B5563;text-transform:uppercase;letter-spacing:1.5px;">Confidence (High)</div>
                            <div style="font-size:2rem;font-weight:800;color:#34D399;font-family:'JetBrains Mono',monospace;">{high_p:.1f}%</div>
                        </div>
                        <div>
                            <div style="font-size:0.6rem;color:#4B5563;text-transform:uppercase;letter-spacing:1.5px;">Confidence (Low)</div>
                            <div style="font-size:2rem;font-weight:800;color:#F87171;font-family:'JetBrains Mono',monospace;">{low_p:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="margin-top:24px;padding:32px;border-radius:20px;
                            background:linear-gradient(135deg,rgba(248,113,113,0.08),rgba(245,158,11,0.05));
                            border:2px solid rgba(248,113,113,0.45);text-align:center;
                            box-shadow:0 0 40px rgba(248,113,113,0.12);">
                    <div style="font-size:3rem;margin-bottom:8px;">⚠️</div>
                    <div style="font-size:2rem;font-weight:800;color:#F87171;letter-spacing:-0.5px;">AT RISK</div>
                    <div style="font-size:0.85rem;color:#4B5563;margin-top:6px;">
                        This student shows behavioural patterns associated with low performance — early intervention recommended
                    </div>
                    <div style="margin-top:20px;display:flex;justify-content:center;gap:32px;">
                        <div>
                            <div style="font-size:0.6rem;color:#4B5563;text-transform:uppercase;letter-spacing:1.5px;">Confidence (Low)</div>
                            <div style="font-size:2rem;font-weight:800;color:#F87171;font-family:'JetBrains Mono',monospace;">{low_p:.1f}%</div>
                        </div>
                        <div>
                            <div style="font-size:0.6rem;color:#4B5563;text-transform:uppercase;letter-spacing:1.5px;">Confidence (High)</div>
                            <div style="font-size:2rem;font-weight:800;color:#34D399;font-family:'JetBrains Mono',monospace;">{high_p:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Probability bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.72rem;color:#4B5563;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;">Probability Distribution</p>', unsafe_allow_html=True)
            st.progress(int(high_p))
            st.caption(f"High: {high_p:.1f}% | Low: {low_p:.1f}%")

            # Input summary expander
            with st.expander("📋  View Input Vector"):
                st.dataframe(input_df, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.error("⚠️  Model file not found. Please run the model training notebook to generate `results/best_model.pkl`.")
    except Exception as e:
        st.error("⚠️  Prediction failed. See details below.")
        with st.expander("Error details"):
            st.exception(e)