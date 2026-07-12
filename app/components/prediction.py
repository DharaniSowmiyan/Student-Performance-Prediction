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
    sub_html = (
        f'<div style="font-size:0.95rem;color:#9CA3AF;margin-top:6px;font-weight:400;line-height:1.5;">{subtitle}</div>'
        if subtitle else ""
    )
    html = (
        '<div style="display:flex;align-items:center;gap:14px;margin:32px 0 20px 0;'
        'padding-bottom:14px;border-bottom:2px solid rgba(56,189,248,0.15);">'
        '<div style="width:42px;height:42px;border-radius:12px;flex-shrink:0;'
        'background:linear-gradient(135deg,#0EA5E9,#6366F1);'
        'display:flex;align-items:center;justify-content:center;font-size:1.1rem;'
        f'box-shadow:0 4px 16px rgba(14,165,233,0.35);">{icon}</div>'
        f'<div><div style="font-size:1.45rem;font-weight:700;color:#F1F5F9;letter-spacing:-0.3px;">{title}</div>'
        f'{sub_html}</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _group_label(text: str):
    st.markdown(
        f'<div style="font-size:0.78rem;font-weight:600;color:#D1D5DB;'
        f'text-transform:uppercase;letter-spacing:1px;margin:18px 0 12px 0;'
        f'padding-left:2px;">{text}</div>',
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pkl_path = os.path.join(base, "results", "best_model.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _slider_label(col: str) -> str:
    return col.replace("_interactions", "").replace("_", " ").strip().title()


def _col_max(col: str) -> int:
    if "clicks" in col:
        return 15000
    if "unique" in col:
        return 50
    if "pre" in col:
        return 800
    if "total" in col:
        return 5000
    return 3000


def _col_default(col: str) -> int:
    if "clicks" in col:
        return 500
    if "unique" in col:
        return 8
    if "pre" in col:
        return 15
    return 50


def _col_step(col: str) -> int:
    mx = _col_max(col)
    if mx >= 5000:
        return 50
    if mx >= 500:
        return 5
    return 1


def show_prediction():
    st.markdown("""
    <div style="padding:8px 0 28px 0;">
        <div style="font-size:0.65rem;color:#6B7280;text-transform:uppercase;
                    letter-spacing:2.5px;margin-bottom:10px;">Real-Time Analysis</div>
        <h1 style="font-size:2.4rem;font-weight:800;margin:0;line-height:1.15;
                   background:linear-gradient(135deg,#F1F5F9 0%,#38BDF8 60%,#6366F1 100%);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Live Prediction
        </h1>
        <p style="color:#9CA3AF;font-size:0.95rem;margin-top:10px;max-width:620px;line-height:1.6;">
            Build a student's LMS behaviour profile using the sliders below.
            The trained Random Forest model will instantly predict their academic outcome.
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

        static_cols  = [c for c in cols if not c.startswith("pat_")]
        pattern_cols = [c for c in cols if c.startswith("pat_")]

        st.markdown(f"""
        <div style="background:rgba(17,24,39,0.75);border:1px solid rgba(56,189,248,0.16);
                    border-radius:16px;padding:18px 24px;margin-bottom:32px;
                    display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:10px;flex:1;min-width:200px;">
                <div style="width:10px;height:10px;border-radius:50%;background:#4ADE80;
                            box-shadow:0 0 8px #4ADE80;flex-shrink:0;"></div>
                <div>
                    <span style="font-size:0.72rem;color:#6B7280;text-transform:uppercase;
                                 letter-spacing:1.5px;">Active Model · </span>
                    <span style="font-size:0.92rem;color:#38BDF8;font-weight:700;">{model_name}</span>
                    <span style="font-size:0.72rem;color:#6B7280;"> · Feature Set: </span>
                    <span style="font-size:0.92rem;color:#A78BFA;font-weight:700;">{feat_set.upper()}</span>
                </div>
            </div>
            <div style="display:flex;gap:24px;">
                <div style="text-align:center;">
                    <div style="font-size:0.62rem;color:#6B7280;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:2px;">F1</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#F1F5F9;
                                font-family:'JetBrains Mono',monospace;">{metrics.get('f1',0):.4f}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:0.62rem;color:#6B7280;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:2px;">Accuracy</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#F1F5F9;
                                font-family:'JetBrains Mono',monospace;">{metrics.get('accuracy',0):.4f}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:0.62rem;color:#6B7280;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:2px;">Precision</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#F1F5F9;
                                font-family:'JetBrains Mono',monospace;">{metrics.get('precision',0):.4f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        _key_kw  = ("total", "unique", "pre")
        key_cols = [c for c in static_cols if any(kw in c.lower() for kw in _key_kw)]
        act_cols = [c for c in static_cols if c not in key_cols]

        with st.form("predict_form"):

            _section(
                "👤", "Student Activity Profile",
                "Use the sliders to set the student's engagement level and interaction counts",
            )

            feature_values: dict = {}

            if key_cols:
                _group_label("Core Engagement Metrics")
                rows_k = [key_cols[i:i+4] for i in range(0, len(key_cols), 4)]
                for row in rows_k:
                    rcols = st.columns(len(row))
                    for idx, c in enumerate(row):
                        with rcols[idx]:
                            feature_values[c] = st.slider(
                                _slider_label(c),
                                min_value=0,
                                max_value=_col_max(c),
                                value=_col_default(c),
                                step=_col_step(c),
                                key=c,
                            )

            if act_cols:
                _group_label("Activity Type Breakdown")
                rows_a = [act_cols[i:i+3] for i in range(0, len(act_cols), 3)]
                for row in rows_a:
                    rcols = st.columns(len(row))
                    for idx, c in enumerate(row):
                        with rcols[idx]:
                            feature_values[c] = st.slider(
                                _slider_label(c),
                                min_value=0,
                                max_value=_col_max(c),
                                value=_col_default(c),
                                step=_col_step(c),
                                key=c,
                            )

            remaining = [c for c in static_cols if c not in feature_values]
            if remaining:
                _group_label("Additional Features")
                rows_r = [remaining[i:i+4] for i in range(0, len(remaining), 4)]
                for row in rows_r:
                    rcols = st.columns(len(row))
                    for idx, c in enumerate(row):
                        with rcols[idx]:
                            feature_values[c] = st.slider(
                                _slider_label(c),
                                min_value=0,
                                max_value=_col_max(c),
                                value=_col_default(c),
                                step=_col_step(c),
                                key=c,
                            )

            if pattern_cols:
                st.markdown("<br>", unsafe_allow_html=True)
                _section(
                    "🔀", "Sequential Behaviour Patterns",
                    "Select the learning navigation sequences observed for this student",
                )

                pat_labels = {c: c.replace("pat_", "").replace("_", " → ") for c in pattern_cols}
                all_labels = list(pat_labels.values())

                selected = st.multiselect(
                    "Observed Navigation Sequences",
                    options=all_labels,
                    default=[],
                    help="Choose every sequence this student has demonstrated. Leave empty if none were recorded.",
                    placeholder="Search or scroll to select sequences…",
                )

                n_sel = len(selected)
                n_tot = len(pattern_cols)
                pill_col = GREEN if n_sel > 3 else ("#FCD34D" if n_sel > 0 else "#6B7280")
                st.markdown(
                    f'<div style="margin-top:8px;">'
                    f'<span style="background:rgba(17,24,39,0.8);'
                    f'border:1px solid rgba(56,189,248,0.15);border-radius:20px;'
                    f'padding:4px 16px;font-size:0.8rem;color:{pill_col};font-weight:600;">'
                    f'{n_sel} of {n_tot} sequences selected'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

                for c in pattern_cols:
                    feature_values[c] = int(pat_labels[c] in selected)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div style="height:1px;background:linear-gradient(90deg,transparent,'
                'rgba(56,189,248,0.22),transparent);margin:4px 0 22px 0;"></div>',
                unsafe_allow_html=True,
            )

            sc1, sc2, sc3 = st.columns([1.5, 1, 1.5])
            with sc2:
                submitted = st.form_submit_button(
                    "⚡  Predict Performance",
                    use_container_width=True,
                )

        if submitted:
            input_df = pd.DataFrame([{c: feature_values[c] for c in cols}])
            pred     = pipeline.predict(input_df)[0]
            proba    = pipeline.predict_proba(input_df)[0]
            high_p   = proba[1] * 100
            low_p    = proba[0] * 100

            if pred == 1:
                st.markdown(f"""
                <div style="margin-top:32px;padding:40px 36px;border-radius:22px;
                            background:linear-gradient(135deg,rgba(52,211,153,0.08),rgba(56,189,248,0.05));
                            border:2px solid rgba(52,211,153,0.45);text-align:center;
                            box-shadow:0 0 48px rgba(52,211,153,0.12);">
                    <div style="font-size:3.2rem;margin-bottom:12px;">🎓</div>
                    <div style="font-size:2.4rem;font-weight:800;color:#34D399;letter-spacing:-0.5px;">HIGH PERFORMER</div>
                    <div style="font-size:0.95rem;color:#9CA3AF;margin-top:10px;
                                max-width:520px;margin-left:auto;margin-right:auto;line-height:1.6;">
                        The model predicts this student is on-track for strong academic performance
                    </div>
                    <div style="margin-top:28px;display:flex;justify-content:center;gap:48px;flex-wrap:wrap;">
                        <div>
                            <div style="font-size:0.65rem;color:#6B7280;text-transform:uppercase;
                                        letter-spacing:1.5px;margin-bottom:4px;">High Performer Confidence</div>
                            <div style="font-size:2.6rem;font-weight:800;color:#34D399;
                                        font-family:'JetBrains Mono',monospace;">{high_p:.1f}%</div>
                        </div>
                        <div>
                            <div style="font-size:0.65rem;color:#6B7280;text-transform:uppercase;
                                        letter-spacing:1.5px;margin-bottom:4px;">At-Risk Probability</div>
                            <div style="font-size:2.6rem;font-weight:800;color:#F87171;
                                        font-family:'JetBrains Mono',monospace;">{low_p:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="margin-top:32px;padding:40px 36px;border-radius:22px;
                            background:linear-gradient(135deg,rgba(248,113,113,0.08),rgba(245,158,11,0.05));
                            border:2px solid rgba(248,113,113,0.45);text-align:center;
                            box-shadow:0 0 48px rgba(248,113,113,0.12);">
                    <div style="font-size:3.2rem;margin-bottom:12px;">⚠️</div>
                    <div style="font-size:2.4rem;font-weight:800;color:#F87171;letter-spacing:-0.5px;">AT RISK</div>
                    <div style="font-size:0.95rem;color:#9CA3AF;margin-top:10px;
                                max-width:520px;margin-left:auto;margin-right:auto;line-height:1.6;">
                        This student shows behavioural patterns associated with low performance —
                        early intervention is recommended
                    </div>
                    <div style="margin-top:28px;display:flex;justify-content:center;gap:48px;flex-wrap:wrap;">
                        <div>
                            <div style="font-size:0.65rem;color:#6B7280;text-transform:uppercase;
                                        letter-spacing:1.5px;margin-bottom:4px;">At-Risk Probability</div>
                            <div style="font-size:2.6rem;font-weight:800;color:#F87171;
                                        font-family:'JetBrains Mono',monospace;">{low_p:.1f}%</div>
                        </div>
                        <div>
                            <div style="font-size:0.65rem;color:#6B7280;text-transform:uppercase;
                                        letter-spacing:1.5px;margin-bottom:4px;">High Performer Confidence</div>
                            <div style="font-size:2.6rem;font-weight:800;color:#34D399;
                                        font-family:'JetBrains Mono',monospace;">{high_p:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📋  View Full Input Vector"):
                st.dataframe(input_df, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.error("⚠️  Model file not found. Please run the model training notebook to generate `results/best_model.pkl`.")
    except Exception as e:
        st.error("⚠️  Prediction failed. See details below.")
        with st.expander("Error details"):
            st.exception(e)
