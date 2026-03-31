import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─── COLOUR PALETTE ───────────────────────────────────────────────────────────
BG       = "#0B0F19"
CARD_BG  = "rgba(17,24,39,0.85)"
PASS_COLOR = "#34D399"
FAIL_COLOR = "#F87171"
DIST_COLOR = "#A78BFA"
WITH_COLOR = "#94A3B8"
ACCENT    = "#38BDF8"

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="'Inter', sans-serif", color="#9CA3AF", size=12),
    margin=dict(l=20, r=20, t=48, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", tickfont=dict(size=11)),
    title_font=dict(size=14, color="#D1D5DB"),
)


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def _section(icon: str, title: str, subtitle: str = ""):
    sub_html = f'<div style="font-size:0.72rem;color:#4B5563;margin-top:1px;">{subtitle}</div>' if subtitle else ""
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
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,0.2),transparent);margin:8px 0;"></div>', unsafe_allow_html=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def show_dashboard():
    # Hero
    st.markdown("""
    <div style="padding:8px 0 24px 0;">
        <div style="font-size:0.65rem;color:#4B5563;text-transform:uppercase;letter-spacing:2.5px;margin-bottom:8px;">
            Open University Learning Analytics Dataset
        </div>
        <h1 style="font-size:2.2rem;font-weight:800;margin:0;line-height:1.2;
                   background:linear-gradient(135deg,#F1F5F9 0%,#38BDF8 60%,#6366F1 100%);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Analytics Dashboard
        </h1>
        <p style="color:#4B5563;font-size:0.85rem;margin-top:6px;">
            Real-time insights from student interaction logs and assessment records
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        with st.spinner("Loading datasets…"):
            student_vle        = pd.read_csv("data/raw/studentVle.csv", nrows=200_000)
            student_info       = pd.read_csv("data/raw/studentInfo.csv")
            vle                = pd.read_csv("data/raw/vle.csv")
            student_assessment = pd.read_csv("data/raw/studentAssessment.csv")
            assessments        = pd.read_csv("data/raw/assessments.csv")

        # ── KPI row ───────────────────────────────────────────────────────────
        _section("📊", "Platform Overview", "Key metrics across the entire dataset")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Students",     f"{student_info['id_student'].nunique():,}")
        c2.metric("Learning Resources", f"{vle['id_site'].nunique():,}")
        c3.metric("VLE Interactions",   f"{len(student_vle):,}+")
        c4.metric("Assessments",        f"{len(assessments):,}")

        _divider()

        # ── Performance ───────────────────────────────────────────────────────
        _section("🎯", "Student Outcome Distribution", "How students finished the course")
        result_counts = student_info["final_result"].value_counts().reset_index()
        result_counts.columns = ["Result", "Count"]
        color_map = {
            "Pass": PASS_COLOR, "Fail": FAIL_COLOR,
            "Distinction": DIST_COLOR, "Withdrawn": WITH_COLOR,
        }

        col_l, col_r = st.columns([3, 2])
        with col_l:
            fig_bar = px.bar(
                result_counts, x="Result", y="Count",
                color="Result", color_discrete_map=color_map, text="Count",
            )
            fig_bar.update_traces(
                texttemplate="%{text:,}", textposition="outside",
                marker_line_width=0, width=0.52,
            )
            fig_bar.update_layout(**CHART_THEME, title="Count by Outcome",
                showlegend=False, yaxis_title="", xaxis_title="")
            fig_bar.update_yaxes(showticklabels=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_r:
            fig_pie = px.pie(
                result_counts, names="Result", values="Count",
                color="Result", color_discrete_map=color_map, hole=0.60,
            )
            fig_pie.update_traces(
                textinfo="percent+label", textfont_size=11,
                pull=[0.03] * len(result_counts),
                marker=dict(line=dict(color=BG, width=3)),
            )
            fig_pie.update_layout(**CHART_THEME, title="Share by Outcome", showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Badges
        rc    = dict(zip(result_counts["Result"], result_counts["Count"]))
        total = result_counts["Count"].sum()
        badge_styles = {
            "Pass":        ("rgba(52,211,153,0.12)", "#34D399", "rgba(52,211,153,0.3)"),
            "Distinction": ("rgba(167,139,250,0.12)","#A78BFA", "rgba(167,139,250,0.3)"),
            "Fail":        ("rgba(248,113,113,0.12)", "#F87171", "rgba(248,113,113,0.3)"),
            "Withdrawn":   ("rgba(148,163,184,0.12)", "#94A3B8", "rgba(148,163,184,0.3)"),
        }
        badges = ""
        for lbl, (bg, fg, bd) in badge_styles.items():
            cnt = rc.get(lbl, 0)
            badges += (f'<span style="display:inline-block;padding:4px 12px;border-radius:20px;'
                       f'font-size:0.7rem;font-weight:600;letter-spacing:0.6px;text-transform:uppercase;'
                       f'margin-right:6px;background:{bg};color:{fg};border:1px solid {bd};">'
                       f'{lbl}&nbsp;{cnt:,} ({cnt/total*100:.1f}%)</span>')
        st.markdown(f'<div style="margin-top:-4px;margin-bottom:12px;">{badges}</div>', unsafe_allow_html=True)

        _divider()

        # ── Activity types ────────────────────────────────────────────────────
        _section("📚", "Learning Resource Types", "Resources available across the platform")
        activity_counts = vle["activity_type"].value_counts().reset_index()
        activity_counts.columns = ["Activity", "Count"]
        fig_act = px.bar(
            activity_counts.sort_values("Count"),
            x="Count", y="Activity", orientation="h",
            color="Count",
            color_continuous_scale=[[0, "#1E3A5F"], [0.5, "#4F46E5"], [1, "#38BDF8"]],
            text="Count",
        )
        fig_act.update_traces(texttemplate="%{text:,}", textposition="outside", marker_line_width=0)
        fig_act.update_coloraxes(showscale=False)
        fig_act.update_layout(**CHART_THEME, title="Resources per Activity Type",
            xaxis_title="", yaxis_title="", height=400)
        fig_act.update_xaxes(showticklabels=False)
        st.plotly_chart(fig_act, use_container_width=True)

        _divider()

        # ── Timeline ──────────────────────────────────────────────────────────
        _section("⏳", "Student Activity Timeline", "Interaction volume by course day")
        timeline = student_vle["date"].value_counts().sort_index().reset_index()
        timeline.columns = ["Day", "Interactions"]
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=timeline["Day"], y=timeline["Interactions"],
            mode="lines",
            line=dict(color=ACCENT, width=2.5),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.06)",
        ))
        fig_line.add_vline(x=0, line_dash="dash", line_color="rgba(248,113,113,0.5)",
            annotation_text="Course Start", annotation_font_color="#F87171",
            annotation_font_size=11)
        fig_line.update_layout(**CHART_THEME,
            title="Daily Interaction Volume (200k sample)",
            xaxis_title="Day Relative to Course Start",
            yaxis_title="Interactions", showlegend=False, height=280)
        st.plotly_chart(fig_line, use_container_width=True)

        _divider()

        # ── Click stats ───────────────────────────────────────────────────────
        _section("🖱️", "Click Behaviour Statistics", "Per-interaction click distribution")
        desc  = student_vle["sum_click"].describe()
        stats = [("Min", f'{desc["min"]:.0f}'), ("25%", f'{desc["25%"]:.0f}'),
                 ("Median", f'{desc["50%"]:.0f}'), ("Mean", f'{desc["mean"]:.1f}'),
                 ("75%", f'{desc["75%"]:.0f}'), ("Max", f'{desc["max"]:.0f}')]
        cols = st.columns(len(stats))
        for col, (label, val) in zip(cols, stats):
            col.markdown(f"""
            <div style="background:rgba(17,24,39,0.7);border:1px solid rgba(56,189,248,0.12);
                        border-radius:12px;padding:14px 16px;text-align:center;">
                <div style="font-size:0.6rem;color:#4B5563;text-transform:uppercase;letter-spacing:1.5px;">{label}</div>
                <div style="font-size:1.25rem;font-weight:700;color:#F1F5F9;font-family:'JetBrains Mono',monospace;">{val}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig_hc = px.histogram(
            student_vle[student_vle["sum_click"] < 50],
            x="sum_click", nbins=49,
            color_discrete_sequence=["#6366F1"],
        )
        fig_hc.update_layout(**CHART_THEME,
            title="Distribution of Clicks per Interaction (clipped ≤50)",
            xaxis_title="Clicks", yaxis_title="Frequency", bargap=0.04)
        st.plotly_chart(fig_hc, use_container_width=True)

        _divider()

        # ── Activity vs Performance ───────────────────────────────────────────
        _section("🔍", "Activity Type vs Student Outcome")
        logs = student_vle.merge(vle, on="id_site", how="left")
        logs = logs.merge(student_info[["id_student", "final_result"]], on="id_student")
        activity_result = (
            logs.groupby(["activity_type", "final_result"])
            .size().reset_index(name="Count")
        )
        fig_grp = px.bar(
            activity_result, x="activity_type", y="Count",
            color="final_result", barmode="group",
            color_discrete_map=color_map,
        )
        fig_grp.update_traces(marker_line_width=0)
        fig_grp.update_layout(**CHART_THEME,
            title="Interactions by Activity Type & Outcome",
            xaxis_title="Activity Type", yaxis_title="Interactions",
            legend_title_text="Outcome", height=400)
        st.plotly_chart(fig_grp, use_container_width=True)

        _divider()

        # ── Assessment scores ─────────────────────────────────────────────────
        _section("📝", "Assessment Score Distribution")
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            fig_sc = px.histogram(
                student_assessment.dropna(subset=["score"]),
                x="score", nbins=50,
                color_discrete_sequence=["#A78BFA"],
            )
            fig_sc.update_layout(**CHART_THEME,
                title="Score Frequency Across All Assessments",
                xaxis_title="Score (0–100)", yaxis_title="Count", bargap=0.04)
            st.plotly_chart(fig_sc, use_container_width=True)

        with col_s2:
            st.markdown('<p style="font-size:.72rem;font-weight:600;color:#4B5563;text-transform:uppercase;letter-spacing:1.4px;margin-bottom:10px;">Score Summary</p>', unsafe_allow_html=True)
            st.dataframe(
                student_assessment["score"].describe().round(2).to_frame().rename(columns={"score": "Value"}),
                use_container_width=True,
            )

        _divider()

        # ── Data Quality ──────────────────────────────────────────────────────
        _section("🧹", "Data Quality — Missing Values")
        datasets = {
            "studentVle": student_vle, "studentInfo": student_info,
            "vle": vle, "studentAssessments": student_assessment, "assessments": assessments,
        }
        missing_data = {n: int(df.isnull().sum().sum()) for n, df in datasets.items()}
        fig_mv = px.bar(
            x=list(missing_data.keys()), y=list(missing_data.values()),
            color=list(missing_data.values()),
            color_continuous_scale=[[0, "#1E3A5F"], [1, "#F87171"]],
            text=list(missing_data.values()),
        )
        fig_mv.update_traces(textposition="outside", marker_line_width=0)
        fig_mv.update_coloraxes(showscale=False)
        fig_mv.update_layout(**CHART_THEME,
            title="Missing Values per Dataset",
            xaxis_title="", yaxis_title="Missing Cells", height=300)
        st.plotly_chart(fig_mv, use_container_width=True)

        _divider()

        # ── Sample data ───────────────────────────────────────────────────────
        _section("🔎", "Sample Interaction Records")
        st.dataframe(student_vle.head(200), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error("⚠️  Could not load datasets. Make sure OULAD CSVs are in `data/raw/`.")
        st.exception(e)