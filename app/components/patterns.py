import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

ACCENT  = "#38BDF8"
GREEN   = "#34D399"
RED     = "#F87171"
PURPLE  = "#A78BFA"
BG_CARD = "rgba(17,24,39,0.85)"

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="'Inter',sans-serif", color="#9CA3AF", size=12),
    margin=dict(l=20, r=20, t=48, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)"),
    title_font=dict(size=14, color="#D1D5DB"),
)


def _section(icon, title, subtitle=""):
    sub_html = f'<div style="font-size:0.72rem;color:#4B5563;">{subtitle}</div>' if subtitle else ""
    html = (
        '<div style="display:flex;align-items:center;gap:12px;margin:28px 0 18px 0;'
        'padding-bottom:12px;border-bottom:1px solid rgba(56,189,248,0.1);">'
        '<div style="width:36px;height:36px;border-radius:10px;'
        'background:linear-gradient(135deg,#0EA5E9,#8B5CF6);'
        'display:flex;align-items:center;justify-content:center;font-size:0.95rem;'
        f'box-shadow:0 3px 12px rgba(14,165,233,0.3);">{icon}</div>'
        '<div>'
        f'<div style="font-size:1rem;font-weight:700;color:#F1F5F9;">{title}</div>'
        f'{sub_html}'
        '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def show_patterns():
    # Hero
    st.markdown("""
    <div style="padding:8px 0 24px 0;">
        <div style="font-size:0.65rem;color:#4B5563;text-transform:uppercase;letter-spacing:2.5px;margin-bottom:8px;">Behavioral Intelligence</div>
        <h1 style="font-size:2.2rem;font-weight:800;margin:0;
                   background:linear-gradient(135deg,#F1F5F9 0%,#A78BFA 60%,#38BDF8 100%);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Pattern Engine
        </h1>
        <p style="color:#4B5563;font-size:0.85rem;margin-top:6px;">
            Sequential activity patterns mined via PrefixSpan — revealing HOW students navigate the LMS
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Resolve paths
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    selected_path = os.path.join(base, "results", "selected_patterns.csv")
    all_path      = os.path.join(base, "results", "patterns.csv")
    pat_fig_path  = os.path.join(base, "results", "figures", "pattern_frequency.png")

    try:
        selected_df = pd.read_csv(selected_path)
        all_df      = pd.read_csv(all_path)

        # ── Summary KPIs ──────────────────────────────────────────────────────
        _section("📊", "Pattern Mining Summary", "Patterns discovered across student cohorts")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Patterns Mined",      f"{len(all_df):,}")
        k2.metric("Discriminative Patterns",   f"{len(selected_df):,}")
        k3.metric("High-Performer Patterns",   f"{(selected_df['group']=='High').sum():,}")
        k4.metric("Low-Performer Patterns",    f"{(selected_df['group']=='Low').sum():,}")

        st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,0.2),transparent);margin:12px 0;"></div>', unsafe_allow_html=True)

        # ── Top discriminative bar chart ───────────────────────────────────
        _section("🔥", "Top Discriminative Patterns", "Highest support difference between High and Low performers")
        top_df = selected_df.copy()
        top_df["abs_diff"] = top_df["difference"].abs()
        top_df = top_df.sort_values("abs_diff", ascending=False).head(12)
        top_df["pattern_label"] = top_df["pattern"].str.replace(",", " →\n", regex=False)

        color_map = {p: GREEN if g == "High" else RED
                     for p, g in zip(top_df["pattern"], top_df["group"])}
        bar_colors = [GREEN if g == "High" else RED for g in top_df["group"]]

        fig = go.Figure()
        for _, row in top_df.iterrows():
            color = GREEN if row["group"] == "High" else RED
            fig.add_trace(go.Bar(
                y=[row["pattern"][:50] + "…" if len(row["pattern"]) > 50 else row["pattern"]],
                x=[row["support_high_pct"] if row["group"] == "High" else row["support_low_pct"]],
                orientation="h",
                name=row["group"],
                marker_color=color,
                marker_line_width=0,
                showlegend=False,
            ))
        fig.update_layout(
            **CHART_THEME,
            title="Pattern Support % by Performance Group",
            xaxis_title="Support (%)",
            yaxis_title="",
            height=520,
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Legend badges
        st.markdown("""
        <div style="margin-top:-8px;margin-bottom:8px;display:flex;gap:10px;">
            <span style="display:inline-flex;align-items:center;gap:6px;background:rgba(52,211,153,0.1);
                         padding:4px 12px;border-radius:20px;border:1px solid rgba(52,211,153,0.3);
                         font-size:0.72rem;color:#34D399;font-weight:600;">
                ● High Performer Patterns
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;background:rgba(248,113,113,0.1);
                         padding:4px 12px;border-radius:20px;border:1px solid rgba(248,113,113,0.3);
                         font-size:0.72rem;color:#F87171;font-weight:600;">
                ● Low Performer Patterns
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,0.2),transparent);margin:12px 0 24px 0;"></div>', unsafe_allow_html=True)

        # ── Side-by-side grouped comparison ───────────────────────────────────
        _section("⚖️", "High vs Low Support Comparison")
        comp_df = selected_df.head(10).copy()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="High Support %",
            x=[p[:40] + "…" if len(p) > 40 else p for p in comp_df["pattern"]],
            y=comp_df["support_high_pct"],
            marker_color=GREEN, marker_line_width=0, opacity=0.88,
        ))
        fig2.add_trace(go.Bar(
            name="Low Support %",
            x=[p[:40] + "…" if len(p) > 40 else p for p in comp_df["pattern"]],
            y=comp_df["support_low_pct"],
            marker_color=RED, marker_line_width=0, opacity=0.88,
        ))
        fig2.update_layout(
            **CHART_THEME, barmode="group",
            title="Support % Comparison — Top 10 Patterns",
            xaxis_title="Pattern", yaxis_title="Support (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(color="#D1D5DB")),
            height=420, xaxis_tickangle=-25,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,0.2),transparent);margin:12px 0 24px 0;"></div>', unsafe_allow_html=True)

        # ── Full table ─────────────────────────────────────────────────────────
        _section("📋", "All Discriminative Patterns", "Browse the complete mined pattern database")

        col_a, col_b = st.columns([1, 2])
        with col_a:
            group_filter = st.selectbox("Filter by Group", ["All", "High", "Low"])
        with col_b:
            search = st.text_input("Search pattern", placeholder="e.g. Discussion, Quiz…")

        display_df = selected_df.copy()
        if group_filter != "All":
            display_df = display_df[display_df["group"] == group_filter]
        if search:
            display_df = display_df[display_df["pattern"].str.contains(search, case=False, na=False)]

        st.dataframe(
            display_df.reset_index(drop=True).style.background_gradient(
                subset=["support_high_pct", "support_low_pct"],
                cmap="Blues",
            ),
            use_container_width=True,
            height=360,
        )
        st.caption(f"Showing {len(display_df)} of {len(selected_df)} discriminative patterns")

        # ── All patterns explorer ──────────────────────────────────────────────
        with st.expander("🔭  Explore ALL Mined Patterns (full database)"):
            sort_col = st.selectbox("Sort by", ["support", "length"], key="all_sort")
            if sort_col in all_df.columns:
                display_all = all_df.sort_values(sort_col, ascending=False).head(200)
            else:
                display_all = all_df.head(200)
            st.dataframe(display_all, use_container_width=True, height=400, hide_index=True)
            st.caption(f"Showing top 200 of {len(all_df):,} total patterns")

    except Exception as e:
        st.error("⚠️  Pattern data not found. Run the pattern mining notebook first.")
        with st.expander("Error details"):
            st.exception(e)