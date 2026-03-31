import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

ACCENT  = "#38BDF8"
GREEN   = "#34D399"
YELLOW  = "#FCD34D"
RED     = "#F87171"
PURPLE  = "#A78BFA"
INDIGO  = "#6366F1"

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="'Inter',sans-serif", color="#9CA3AF", size=12),
    margin=dict(l=20, r=20, t=48, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)"),
    title_font=dict(size=14, color="#D1D5DB"),
)

MODEL_COLORS = {
    "RandomForest":       "#34D399",
    "LightGBM":           "#38BDF8",
    "XGBoost":            "#FCD34D",
    "SVM":                "#A78BFA",
    "LogisticRegression": "#F87171",
}


def _hex_to_rgba(hex_color: str, alpha: float = 0.07) -> str:
    """Convert #RRGGBB hex color to rgba(r,g,b,alpha) string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


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


def _divider():
    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,'
        'rgba(56,189,248,0.2),transparent);margin:8px 0 20px 0;"></div>',
        unsafe_allow_html=True,
    )


def show_model_results():
    st.markdown("""
    <div style="padding:8px 0 24px 0;">
        <div style="font-size:0.65rem;color:#4B5563;text-transform:uppercase;letter-spacing:2.5px;margin-bottom:8px;">Experiment Results</div>
        <h1 style="font-size:2.2rem;font-weight:800;margin:0;
                   background:linear-gradient(135deg,#F1F5F9 0%,#34D399 50%,#38BDF8 100%);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Model Comparison
        </h1>
        <p style="color:#4B5563;font-size:0.85rem;margin-top:6px;">
            5-fold stratified cross-validation results across 5 models × 3 feature sets
        </p>
    </div>
    """, unsafe_allow_html=True)

    base         = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    results_path = os.path.join(base, "results", "model_results.csv")
    best_path    = os.path.join(base, "results", "best_model_info.csv")
    fig_dir      = os.path.join(base, "results", "figures")

    try:
        results  = pd.read_csv(results_path)
        best_row = pd.read_csv(best_path).iloc[0]

        # ── Champion card ──────────────────────────────────────────────────────
        _section("🏆", "Best Model", "Top performer across all experiments")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(52,211,153,0.08),rgba(56,189,248,0.06));
                    border:1px solid rgba(52,211,153,0.3);border-radius:18px;padding:24px 28px;
                    margin-bottom:20px;display:flex;align-items:center;gap:24px;">
            <div style="font-size:2.8rem;">🥇</div>
            <div>
                <div style="font-size:1.5rem;font-weight:800;color:#34D399;">{best_row['model']}</div>
                <div style="font-size:0.8rem;color:#4B5563;margin-top:2px;">Feature Set: <span style="color:#38BDF8;font-weight:600;">{best_row['features'].upper()}</span></div>
            </div>
            <div style="margin-left:auto;display:flex;gap:20px;text-align:center;">
                <div>
                    <div style="font-size:0.6rem;color:#4B5563;text-transform:uppercase;letter-spacing:1.5px;">F1 Score</div>
                    <div style="font-size:1.8rem;font-weight:800;color:#F1F5F9;font-family:'JetBrains Mono',monospace;">{best_row['f1']:.4f}</div>
                </div>
                <div>
                    <div style="font-size:0.6rem;color:#4B5563;text-transform:uppercase;letter-spacing:1.5px;">Accuracy</div>
                    <div style="font-size:1.8rem;font-weight:800;color:#F1F5F9;font-family:'JetBrains Mono',monospace;">{best_row['accuracy']:.4f}</div>
                </div>
                <div>
                    <div style="font-size:0.6rem;color:#4B5563;text-transform:uppercase;letter-spacing:1.5px;">Precision</div>
                    <div style="font-size:1.8rem;font-weight:800;color:#F1F5F9;font-family:'JetBrains Mono',monospace;">{best_row['precision']:.4f}</div>
                </div>
                <div>
                    <div style="font-size:0.6rem;color:#4B5563;text-transform:uppercase;letter-spacing:1.5px;">Recall</div>
                    <div style="font-size:1.8rem;font-weight:800;color:#F1F5F9;font-family:'JetBrains Mono',monospace;">{best_row['recall']:.4f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        _divider()

        # ── Full Leaderboard ───────────────────────────────────────────────────
        _section("📋", "Full Leaderboard", "All 15 experiments sorted by F1 Score")
        leaderboard = results.copy()
        leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))
        st.dataframe(
            leaderboard.style.format({
                "f1": "{:.4f}", "accuracy": "{:.4f}",
                "precision": "{:.4f}", "recall": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        _divider()

        # ── F1 Grouped bar chart ───────────────────────────────────────────────
        _section("📊", "F1 Score by Model & Feature Set", "Grouped comparison across all configurations")
        models_order = ["LogisticRegression", "SVM", "XGBoost", "LightGBM", "RandomForest"]
        feat_sets    = ["static", "sequence", "hybrid"]
        fig_f1 = go.Figure()
        for feat in feat_sets:
            sub  = results[results["features"] == feat].set_index("model")
            vals = [sub.loc[m, "f1"] if m in sub.index else 0 for m in models_order]
            fig_f1.add_trace(go.Bar(
                name=feat.upper(),
                x=models_order, y=vals,
                marker_line_width=0, opacity=0.88,
                text=[f"{v:.3f}" for v in vals],
                textposition="outside",
                textfont=dict(size=10, color="#D1D5DB"),
            ))
        fig_f1.update_layout(
            **CHART_THEME, barmode="group",
            title="F1 Score by Model and Feature Set",
            xaxis_title="", yaxis_title="F1 Score (weighted)",
            yaxis_range=[0, 1.0],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(color="#D1D5DB")),
            height=420,
            colorway=["#6366F1", "#38BDF8", "#34D399"],
        )
        st.plotly_chart(fig_f1, use_container_width=True)

        _divider()

        # ── Radar chart ────────────────────────────────────────────────────────
        _section("🕸️", "Multi-Metric Radar", "Compare models across all 4 metrics")
        metrics      = ["accuracy", "precision", "recall", "f1"]
        radar_models = results.groupby("model")[metrics].max().reset_index()
        fig_radar    = go.Figure()
        for _, row in radar_models.iterrows():
            color = MODEL_COLORS.get(row["model"], "#94A3B8")
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics] + [row[metrics[0]]],
                theta=["Accuracy", "Precision", "Recall", "F1", "Accuracy"],
                name=row["model"],
                line_color=color,
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.07),
            ))
        fig_radar.update_layout(
            **CHART_THEME,
            polar=dict(
                radialaxis=dict(visible=True, range=[0.6, 0.85], color="#4B5563",
                                gridcolor="rgba(255,255,255,0.05)"),
                angularaxis=dict(color="#9CA3AF"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, font=dict(color="#D1D5DB")),
            height=450,
            title_text="Best Metrics per Model (max across feature sets)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        _divider()

        # ── F1 Heatmap ─────────────────────────────────────────────────────────
        _section("🗺️", "F1 Score Heatmap", "Feature set × model performance overview")
        pivot  = results.pivot_table(index="model", columns="features", values="f1")
        fig_hm = px.imshow(
            pivot,
            color_continuous_scale=[[0, "#0D1321"], [0.4, "#0EA5E9"], [1, "#34D399"]],
            text_auto=".3f",
            aspect="auto",
            zmin=0.60,
            zmax=0.85,
        )
        fig_hm.update_traces(textfont=dict(size=14, color="white"))
        fig_hm.update_layout(
            **CHART_THEME,
            title="F1 Score Heatmap — Model × Feature Set",
            height=350,
            coloraxis_colorbar=dict(tickfont=dict(color="#9CA3AF")),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        _divider()

        # ── Static figures ─────────────────────────────────────────────────────
        _section("🖼️", "Generated Charts", "High-quality charts from the analysis pipeline")
        feat_imp  = os.path.join(fig_dir, "feature_importance.png")
        confusion = os.path.join(fig_dir, "confusion_matrix.png")

        cols_fig = st.columns(2)
        if os.path.exists(feat_imp):
            with cols_fig[0]:
                st.markdown(
                    '<p style="font-size:.75rem;color:#4B5563;text-transform:uppercase;'
                    'letter-spacing:1.5px;margin-bottom:8px;">Feature Importance</p>',
                    unsafe_allow_html=True,
                )
                st.image(feat_imp, use_container_width=True)
        if os.path.exists(confusion):
            with cols_fig[1]:
                st.markdown(
                    '<p style="font-size:.75rem;color:#4B5563;text-transform:uppercase;'
                    'letter-spacing:1.5px;margin-bottom:8px;">Confusion Matrix</p>',
                    unsafe_allow_html=True,
                )
                st.image(confusion, use_container_width=True)

    except Exception as e:
        st.error("⚠️  Model results not found. Run the model training notebook first.")
        with st.expander("Error details"):
            st.exception(e)
