import os
import base64
from typing import Tuple 

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

#page setup

st.set_page_config(
    page_title="IISc Research EDA",
    page_icon="📊",
    layout="wide",
)

#style

CUSTOM_CSS = """
<style>
/* Main Background: Dark Navy */
html, body, .stApp {
    background-color: #1e293b !important;
    font-family: "Inter", sans-serif;
    color: #ffffff; 
}

/* Sidebar: Slightly darker shade for contrast */
[data-testid="stSidebar"] {
    background: #0f172a !important; 
    color: #ffffff !important;
}

/* KPI Card Styling (kept minimalist) */
.metric-card {
    background: #334155 !important;
    border-radius: 10px;
    padding: 1.5rem; 
    border: 1px solid #475569; 
    color: #ffffff !important;
    min-height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: space-between; 
}

/* KPI Title Accent */
.kpi-section-title {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #facc15 !important; 
    margin-top: 1.0rem !important; 
    margin-bottom: 1.0rem !important;
}

/* KPI Value & Subtext */
.metric-value {
    font-size: 2.0rem !important; 
    font-weight: 800 !important;
    color: #ffffff !important;
}
.metric-change {
    font-weight: 700;
}
.metric-change.positive {
    color: #10b981 !important; 
}
.metric-change.negative {
    color: #facc15 !important; 
}

/* Insight Headers */
.insights-header {
    color: #ffffff !important;
    font-size: 1.2rem;
    margin-bottom: 1rem;
}

/* --- MODIFIED STYLE FOR ST.CODE BLOCKS (Matches card background) --- */
div[data-testid="stCodeBlock"] {
    background: #334155 !important; /* Match card background */
    border: 1px solid #475569 !important; /* Match card border */
    border-radius: 10px; 
    color: #ffffff !important; 
    padding: 1rem; 
    font-family: "Inter", sans-serif; 
    font-size: 0.9rem; 
}

/* Ensure pre-formatted text inside the code block is white */
div[data-testid="stCodeBlock"] pre {
    color: #ffffff !important;
    background: transparent !important; 
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)



# Functions

def get_logo_b64(path: str = "assets/IISc_Master_Seal.jpg") -> str:
    """Load IISc logo."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except: 
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


@st.cache_data(show_spinner=True)
def get_research_data(path: str = "data/publications_cleaned.csv") -> pd.DataFrame:
    """Load data (Uses simple mock data if file not found)."""
    try:
        df_local = pd.read_csv(path)
    except FileNotFoundError:
        
        years = list(range(2000, 2024))
        num_years = len(years)
        np.random.seed(42) 
        
        base_docs = 10000 + np.arange(num_years) * 800 + np.random.randint(-500, 500, num_years)
        
        data = {
            'year': years,
            'WoS Documents': base_docs.astype(int),
            'Times Cited': (base_docs * 10 + np.random.randint(-1000, 1000, num_years)).astype(int),
            'CNCI': 1.0 + np.random.uniform(-0.1, 0.1, num_years),
            'Documents in Top 10%': (base_docs * 0.1).astype(int),
            'Documents in Top 1%': (base_docs * 0.01).astype(int),
        }
        df_local = pd.DataFrame(data).sort_values("year")

        # Calculated all dependent columns
        df_local['Citation per Document'] = df_local['Times Cited'] / df_local['WoS Documents']
        df_local['Top 10% Docs %'] = (df_local['Documents in Top 10%'] / df_local['WoS Documents']) * 100
        df_local['Top 1% Docs %'] = (df_local['Documents in Top 1%'] / df_local['WoS Documents']) * 100
        
        # dummy columns 
        df_local["Collab-CNCI"] = df_local['CNCI'] + 0.1
        df_local["Docs Cited %"] = 90.0
        df_local["Rank"] = 1
        df_local["Top 10% Contribution Rate"] = 1.0
        df_local["Top 1% Contribution Rate"] = 1.0
        
    df_local.columns = [c.strip() for c in df_local.columns] 
    
    
    for col in df_local.columns:
         if 'year' not in col.lower():
             df_local[col] = pd.to_numeric(df_local[col], errors="coerce")
             
    df_local = df_local.dropna(subset=["year"]).sort_values("year")
    return df_local



def format_big_num(n: float) -> str:
    """Quick formatting for big numbers (K/M)."""
    if pd.isna(n): return "-"
    n_abs = abs(n)
    if n_abs >= 1_000_000: return f"{n/1_000_000:.2f}M"
    if n_abs >= 1_000: return f"{n/1_000:.1f}K"
    return f"{n:,.0f}"


def calculate_yoy_percent(df: pd.DataFrame, col: str) -> Tuple[float, int]:
    """YoY % change."""
    df_sorted = df.sort_values("year")
    df_filtered = df_sorted[df_sorted[col].notna()].tail(2)

    if len(df_filtered) < 2:
        return np.nan, int(df_sorted["year"].max()) if not df_sorted.empty else 0
    
    last_two = df_filtered.tail(2)
    prev_val = last_two.iloc[0][col]
    curr_val = last_two.iloc[1][col]
    
    if prev_val == 0 or pd.isna(prev_val) or pd.isna(curr_val):
        return np.nan, int(last_two.iloc[1]["year"])
    
    return (curr_val - prev_val) / prev_val * 100.0, int(last_two.iloc[1]["year"])


df = get_research_data("data/publications_cleaned.csv")
logo_b64 = get_logo_b64("assets/IISc_Master_Seal.jpg")


with st.sidebar:
    st.markdown("## 📊 Research Data Explorer")
    st.caption("PAIU-OPSA · IISc Bangalore")

    years = sorted(df["year"].unique().astype(int))
    min_y, max_y = int(min(years)), int(max(years))

    year_range = st.slider(
        "Select Year Range",
        min_value=min_y,
        max_value=max_y,
        value=(min_y, max_y),
        step=1,
    )

    trend_metric = st.selectbox(
        "Metric Trend",
        ["WoS Documents", "Times Cited", "CNCI", "Citation per Document"],
        index=0,
    )

    st.markdown("---")
    show_raw = st.checkbox("Show raw data", False)
    show_corr = st.checkbox("Show correlation heatmap", False)

    st.markdown("---")
    st.markdown(
        '<div class="info-box">'
        "Use filters to focus on growth or decline phases. Check the documentation for metric definitions."
        "</div>", 
        unsafe_allow_html=True,
    )


mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
df_f = df[mask].copy()
if df_f.empty:
    st.error("No data in this year range. Adjust the slider.")
    st.stop()

# --- HEADER ---

st.markdown(
    f"""
    <div style='text-align: center; margin-bottom: 2.0rem;'>
        <img src="data:image/jpeg;base64,{logo_b64}" width="55" height="55" style="vertical-align: middle; margin-right: 10px;"/>
        <span style='font-size: 2.0rem; font-weight: 800; color: #facc15;'>
            IISc Research Metrics Dashboard
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)



st.markdown(
    "<div class='kpi-section-title'>Key Performance Indicators (KPIs) Overview</div>",
    unsafe_allow_html=True,
)

total_docs = df_f["WoS Documents"].sum()
total_cites = df_f["Times Cited"].sum()
avg_cnci = df_f["CNCI"].mean()
avg_cpd = df_f["Citation per Document"].mean()


chg_docs, latest_year_docs = calculate_yoy_percent(df_f, "WoS Documents") 
chg_cites, _ = calculate_yoy_percent(df_f, "Times Cited")
chg_cnci, _ = calculate_yoy_percent(df_f, "CNCI")
chg_cpd, _ = calculate_yoy_percent(df_f, "Citation per Document")

c1, c2, c3, c4 = st.columns(4)


with c1:
    docs_change_text = f"{chg_docs:+.1f}%" if pd.notna(chg_docs) else "N/A"
    docs_change_class = "positive" if pd.notna(chg_docs) and chg_docs > 0 else "negative" if pd.notna(chg_docs) and chg_docs < 0 else ""
    docs_arrow = '▲' if pd.notna(chg_docs) and chg_docs > 0 else '▼' if pd.notna(chg_docs) and chg_docs < 0 else '•'
    
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">TOTAL WOS DOCUMENTS</div>
          <div class="metric-value">{format_big_num(total_docs)}</div>
          <div class="metric-subtext">Cumulative Output</div>
          <div class="metric-change {docs_change_class}">
              {docs_arrow} YoY: {docs_change_text}
          </div>
        </div>
        """, unsafe_allow_html=True
    )


with c2:
    cites_change_text = f"{chg_cites:+.1f}%" if pd.notna(chg_cites) else "N/A"
    cites_change_class = "positive" if pd.notna(chg_cites) and chg_cites > 0 else "negative" if pd.notna(chg_cites) and chg_cites < 0 else ""
    cites_arrow = '▲' if pd.notna(chg_cites) and chg_cites > 0 else '▼' if pd.notna(chg_cites) and chg_cites < 0 else '•'
    
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">TOTAL CITATIONS</div>
          <div class="metric-value">{format_big_num(total_cites)}</div>
          <div class="metric-subtext">Total Impact Tally</div>
          <div class="metric-change {cites_change_class}">
              {cites_arrow} YoY: {cites_change_text}
          </div>
        </div>
        """, unsafe_allow_html=True
    )


with c3:
    cnci_change_text = f"{chg_cnci:+.1f}%" if pd.notna(chg_cnci) else "N/A"
    cnci_change_class = "positive" if pd.notna(chg_cnci) and chg_cnci > 0 else "negative" if pd.notna(chg_cnci) and chg_cnci < 0 else ""
    cnci_arrow = '▲' if pd.notna(chg_cnci) and chg_cnci > 0 else '▼' if pd.notna(chg_cnci) and chg_cnci < 0 else '•'
    
    cnci_status = 'Above World Avg (1.0)' if avg_cnci >= 1.0 else 'Below World Avg (1.0)'
    
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">AVERAGE CNCI</div>
          <div class="metric-value">{avg_cnci:.2f}</div>
          <div class="metric-subtext">{cnci_status}</div>
          <div class="metric-change {cnci_change_class}">
              {cnci_arrow} YoY: {cnci_change_text}
          </div>
        </div>
        """, unsafe_allow_html=True
    )


with c4:
    cpd_change_text = f"{chg_cpd:+.1f}%" if pd.notna(chg_cpd) else "N/A"
    cpd_change_class = "positive" if pd.notna(chg_cpd) and chg_cpd > 0 else "negative" if pd.notna(chg_cpd) and chg_cpd < 0 else ""
    cpd_arrow = '▲' if pd.notna(chg_cpd) and chg_cpd > 0 else '▼' if pd.notna(chg_cpd) and chg_cpd < 0 else '•'
    
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">CITATIONS PER DOCUMENT</div>
          <div class="metric-value">{avg_cpd:.1f}</div>
          <div class="metric-subtext">Depth of Citation Impact</div>
          <div class="metric-change {cpd_change_class}">
              {cpd_arrow} YoY: {cpd_change_text}
          </div>
        </div>
        """, unsafe_allow_html=True
    )

st.write("") 
st.markdown("---") 

# TREND CHART 

trend_line_fig = px.line(
    df_f,
    x="year",
    y=trend_metric,
    markers=True,
    template="plotly_dark", 
    color_discrete_sequence=['#facc15'] 
)
trend_line_fig.update_layout(
    title={
        'text': f"Annual Trend: {trend_metric} over Time",
        'y':0.95, 
        'x':0.5,
        'xanchor': 'center',
        'font': {'size': 16, 'color': '#ffffff'}
    }, 
    height=360,
    margin=dict(t=40, b=0), 
    plot_bgcolor='#334155', 
    paper_bgcolor='#334155', 
    font_color="#ffffff"
)
trend_line_fig.update_xaxes(showgrid=False)
st.plotly_chart(trend_line_fig, use_container_width=True)


st.write("")

# DONUT + BAR

col_donut, col_bar = st.columns([1.1, 1.9])


with col_donut:

    total_docs_sel = df_f["WoS Documents"].sum()
    docs_top10 = df_f["Documents in Top 10%"].sum()
    docs_top1 = df_f["Documents in Top 1%"].sum()
    docs_other = total_docs_sel - docs_top10 

    labels = ["Other docs", "Top 10% docs (Exc. Top 1%)", "Top 1% docs"] 
    values = [max(docs_other, 0), max(docs_top10 - docs_top1, 0), docs_top1] 
    
    colors = ['#475569', '#facc15', '#ffffff'] 

    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=labels, 
                values=values, 
                hole=0.6,
                marker_colors=colors,
                name="Impact Tiers",
                hovertemplate='%{label}<br>Documents: %{value} <extra></extra>' 
            )
        ]
    )
    fig_donut.update_layout(
        title={
            'text': f"Total Documents by Citation Tier", 
            'y':0.95, 
            'x':0.5,
            'xanchor': 'center',
            'font': {'size': 15, 'color': '#ffffff'}
        }, 
        height=360, 
        margin=dict(t=40, b=0), 
        showlegend=True,
        template="plotly_dark",
        plot_bgcolor='#334155', 
        paper_bgcolor='#334155', 
        font_color="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_donut, use_container_width=True)


with col_bar:

    if "Top 10% Docs %" in df_f.columns:
        fig_bar = px.bar(
            df_f,
            x="year",
            y="Top 10% Docs %",
            template="plotly_dark",
            color_discrete_sequence=['#facc15'] 
        )
        fig_bar.update_layout(
            title={
                'text': f"Annual % of Docs in Top 10%",
                'y':0.95, 
                'x':0.5,
                'xanchor': 'center',
                'font': {'size': 15, 'color': '#ffffff'}
            }, 
            xaxis_title="Year",
            yaxis_title="% Top 10%", 
            height=360, 
            margin=dict(t=40, b=0), 
            plot_bgcolor='#334155', 
            paper_bgcolor='#334155', 
            font_color="#ffffff"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Top 10% data column missing.")


#INSIGHTS + RECENT TRAJECTORY

st.markdown("---") 

col_hi, col_recent = st.columns(2)



with col_hi:
    st.markdown("<h3 class='insights-header'>Peak Performance Highlights</h3>", unsafe_allow_html=True) 
    
    # 1. Find the best values
    best_cnci_row = df_f.loc[df_f["CNCI"].idxmax()] if "CNCI" in df_f.columns and not df_f["CNCI"].empty else None
    best_cnci_year = int(best_cnci_row["year"]) if best_cnci_row is not None and pd.notna(best_cnci_row["year"]) else "N/A"
    best_cnci_val = best_cnci_row["CNCI"] if best_cnci_row is not None else 0.0

    best_docs_row = df_f.loc[df_f["WoS Documents"].idxmax()] if "WoS Documents" in df_f.columns and not df_f["WoS Documents"].empty else None
    best_docs_year = int(best_docs_row["year"]) if best_docs_row is not None and pd.notna(best_docs_row["year"]) else "N/A"
    best_docs_val = best_docs_row["WoS Documents"] if best_docs_row is not None else 0

    best_top10_row = df_f.loc[df_f["Top 10% Docs %"].idxmax()] if "Top 10% Docs %" in df_f.columns and not df_f["Top 10% Docs %"].empty else None
    best_top10_year = int(best_top10_row["year"]) if best_top10_row is not None and pd.notna(best_top10_row["year"]) else "N/A"
    best_top10_val = best_top10_row["Top 10% Docs %"] if best_top10_row is not None else 0.0

    
    report_content = f"""
HIGHLIGHTS (Range: {year_range[0]} - {year_range[1]})
------------------------------------------------------
1. Highest Normalized Impact (CNCI):
   Value: {best_cnci_val:.2f}
   Year:  {best_cnci_year}

2. Peak Publication Volume:
   Value: {format_big_num(best_docs_val)} documents
   Year:  {best_docs_year}

3. Best Top 10% Share:
   Value: {best_top10_val:.2f}% of total documents
   Year:  {best_top10_year}

"""
    
 
    st.code(report_content, language='text')




with col_recent:
    st.markdown("<h3 class='insights-header'>Recent Trajectory (Latest Annual Metrics)</h3>", unsafe_allow_html=True) 
    
    df_sorted = df_f.sort_values("year")
    latest_year = int(df_sorted["year"].max()) if not df_sorted.empty and pd.notna(df_sorted["year"].max()) else "N/A"
    
    # Get latest values for display
    latest = df_sorted[df_sorted["year"] == latest_year].iloc[0] if isinstance(latest_year, int) else None
    latest_docs = latest["WoS Documents"] if latest is not None and "WoS Documents" in latest else 0
    latest_cites = latest["Times Cited"] if latest is not None and "Times Cited" in latest else 0
    latest_cnci = latest["CNCI"] if latest is not None and "CNCI" in latest else 0.0

   
    def format_change_text(change):
        if pd.isna(change):
            return "N/A"
       
        sign = '▲' if change >= 0 else '▼'
        return f"{sign} {change:+.1f}%" 

    
    docs_change_text = format_change_text(chg_docs)
    cites_change_text = format_change_text(chg_cites)
    cnci_change_text = format_change_text(chg_cnci)

    
    trajectory_content = f"""
TRAJECTORY (Latest Year: {latest_year})
------------------------------------------------------
1. Latest Output:
   Value: {format_big_num(latest_docs)} documents
   YoY Change: {docs_change_text}
   

2. Latest Citations:
   Value: {format_big_num(latest_cites)} citations
   YoY Change: {cites_change_text}

3. Latest CNCI:
   Value: {latest_cnci:.2f}
   YoY Change: {cnci_change_text}

"""

    
    st.code(trajectory_content, language='text')




if show_corr:
    st.markdown("---") 

    corr_cols = [
        "WoS Documents",
        "Times Cited",
        "CNCI",
        "Citation per Document",
        "Docs Cited %",
        "Top 10% Docs %",
        "Top 1% Docs %",
    ]
    valid_corr_cols = [col for col in corr_cols if col in df_f.columns]
    
    if len(valid_corr_cols) >= 2:
        corr_df = df_f[valid_corr_cols].corr().round(2)
        fig_corr = px.imshow(
            corr_df,
            text_auto=True, 
            color_continuous_scale=[
                [0.0, "rgb(178,24,43)"],
                [0.5, "rgb(247,247,247)"],
                [1.0, "#facc15"]
            ], 
            range_color=[-1, 1],
            aspect="auto",
            template="plotly_dark"
        )
        fig_corr.update_layout(
            title={
                'text': f"Correlation Heatmap of Key Metrics",
                'y':0.95, 
                'x':0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': '#ffffff'}
            }, 
            height=400, 
            margin=dict(t=40, b=0), 
            plot_bgcolor='#334155', 
            paper_bgcolor='#334155', 
            font_color="#ffffff"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")



if show_raw:
    st.markdown("---")
    st.markdown("### Filtered Data")
    st.dataframe(df_f, use_container_width=True)

    csv_bytes = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Data (CSV)",
        data=csv_bytes,
        file_name="iisc_research_filtered.csv",
        mime="text/csv",
    )