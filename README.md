📝 IISc Research EDA Dashboard: Developer Notes
Status: V3.0 (Themed Text Boxes, Dark Mode Finalized)

This is a Streamlit app built for PAIU-OPSA at IISc. It's designed to give quick, high-level Exploratory Data Analysis (EDA) on our core Web of Science (WoS) metrics (Documents, Citations, CNCI). It uses a custom dark theme for the executive summary view.

🛠️ Core Functionality
KPIs at a Glance: The top row calculates and displays the current totals and averages (Documents, Citations, CNCI, CPD) based on the filter, including a Yo-Y % change calculation using the last two years of the filtered set.

Flexible Filtering: We use the sidebar st.slider to dynamically change the date range (df_f is the filtered DataFrame).

Trend Chart: The main chart is a line plot showing the trend of a user-selected metric (default is WoS Documents) over the filtered years.

Impact Metrics Deep Dive:

Tier Distribution: A donut chart showing the split between documents in the Top 1%, Top 10% (excluding Top 1%), and the rest.

Annual Quality: A bar chart tracking the Top 10% Docs % year-by-year.

Insight Boxes (The st.code blocks):

Peak Highlights: A single box summarizing the best year for CNCI, Publications, and Top 10% share across the entire filtered range.

Recent Trajectory: Individual, card-like boxes (now using custom HTML/CSS for polish) showing the latest year's value and its YoY change for Output, Citations, and CNCI.

Debugging/Exploration Tools: Toggles for a Correlation Heatmap and the Raw Data table are included in the sidebar.

Styling: Custom CSS is embedded (CUSTOM_CSS variable) to enforce a dark blue/navy background (#1e293b) and a gold accent (#facc15).