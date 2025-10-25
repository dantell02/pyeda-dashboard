"""
Streamlit page for the 'Global Cost Overview' section of the dashboard.
Provides visualizations including a choropleth map, a dynamic ranking chart,
and a cost composition sunburst chart.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import pycountry
import numpy as np
from pathlib import Path
import sys

# --- Data Loading ---
# Add parent directory to path to import preprocessing module.
try:
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
    from preprocessing import load_and_preprocess_education_data
except (ModuleNotFoundError, ImportError):
    st.error(
        "Could not find the 'preprocessing.py' module. Please check if it's in the 'src/' directory."
    )
    st.stop()


@st.cache_data
def load_data():
    """Loads and caches the preprocessed education data."""
    return load_and_preprocess_education_data()


df = load_data()

# --- Data Validation ---
# Ensures that essential columns are present in the loaded dataframe.
required_cols = {
    "country", "level", "field_of_study",
    "total_annual_cost", "tuition_usd", "rent_usd",
    "insurance_usd", "visa_fee_usd"
}
if not required_cols.issubset(df.columns):
    missing = required_cols - set(df.columns)
    st.error(f"Missing required columns in the preprocessed data: {', '.join(sorted(missing))}")
    st.stop()

# --- Feature Engineering: Region Mapping ---
# Creates a 'region' column for broader geographical analysis.
REGION_MAP = {
    "USA": "North America", "Canada": "North America", "UK": "Europe",
    "Germany": "Europe", "Netherlands": "Europe", "Ireland": "Europe",
    "Turkey": "Europe", "UAE": "Middle East", "Australia": "Oceania",
    "New Zealand": "Oceania", "Singapore": "Asia", "South Korea": "Asia",
    "Hong Kong": "Asia",
}
df["region"] = df["country"].map(REGION_MAP).fillna("Other")

# --- Page Layout ---
st.title("Global Cost Overview")
st.caption("Explore the costs of education with interactive maps and charts.")

# Display high-level KPI cards.
with st.container():
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Countries", df["country"].nunique())
    k2.metric("Fields of Study", df["field_of_study"].nunique())
    k3.metric("Median Total Cost", f"${df['total_annual_cost'].median():,.0f}")
    k4.metric("Mean Tuition", f"${df['tuition_usd'].mean():,.0f}")

# Create navigable tabs for each plot.
tab1, tab2, tab3 = st.tabs(["🌍Global Map", "🏆Country Ranker", "📊Cost Composition"])

# --- TAB 1: Global Map ---
with tab1:
    st.header("Global Cost Overview Map")

    map_cost_metric = st.selectbox(
        "Select a cost metric to display on the map:",
        options=[
            ('total_annual_cost', 'Total Annual Cost'),
            ('tuition_usd', 'Tuition'),
            ('rent_usd', 'Monthly Rent'),
            ('insurance_usd', 'Insurance'),
            ('visa_fee_usd', 'Visa Fee')
        ],
        format_func=lambda x: x[1]
    )

    use_log_scale = st.toggle("Use Logarithmic Scale for Colors", value=True, help="Use a log scale to better distinguish between lower-cost countries.")

    # Aggregate data for the map.
    country_costs = df.groupby('country')[map_cost_metric[0]].mean().reset_index()

    @st.cache_data
    def get_iso_alpha(country_name):
        """Converts country names to ISO alpha-3 codes for mapping."""
        country_map = {"USA": "USA", "UK": "GBR", "South Korea": "KOR", "UAE": "ARE", "Turkey": "TUR"}
        if country_name in country_map:
            return country_map[country_name]
        try:
            return pycountry.countries.search_fuzzy(country_name)[0].alpha_3
        except LookupError:
            return None

    country_costs['iso_alpha'] = country_costs['country'].apply(get_iso_alpha)
    country_costs.dropna(subset=['iso_alpha'], inplace=True)

    # Determine which column to use for color based on the toggle.
    if use_log_scale:
        color_col = f"log_{map_cost_metric[0]}"
        country_costs[color_col] = np.log10(country_costs[map_cost_metric[0]])
    else:
        color_col = map_cost_metric[0]

    fig_map = px.choropleth(
        country_costs,
        locations="iso_alpha",
        color=color_col,
        hover_name="country",
        custom_data=['country', map_cost_metric[0]],
        color_continuous_scale=px.colors.sequential.Plasma,
        title=f"Global Map of {map_cost_metric[1]}"
    )
    fig_map.update_traces(
        hovertemplate='<b>%{customdata[0]}</b><br><br>' + f'{map_cost_metric[1]}: $'+'%{customdata[1]:,.2f}' + '<extra></extra>'
    )

    # Dynamically adjust color bar layout based on scale selection.
    if use_log_scale:
        if map_cost_metric[0] == 'rent_usd':
            tick_values = [500, 750, 1000, 1500, 2000, 2500]
        elif map_cost_metric[0] in ['insurance_usd', 'visa_fee_usd']:
             tick_values = [50, 100, 200, 500, 1000, 1500]
        else:
            tick_values = [2000, 5000, 10000, 20000, 50000, 100000]
        tick_text = [f"${val:,}" for val in tick_values]
        fig_map.update_layout(
            geo=dict(showframe=False, showcoastlines=False, projection_type='natural earth'),
            margin={"r":0,"t":40,"l":0,"b":0},
            coloraxis_colorbar=dict(title=f"{map_cost_metric[1]} (USD)", tickvals=np.log10(tick_values), ticktext=tick_text)
        )
    else:
        fig_map.update_layout(
            geo=dict(showframe=False, showcoastlines=False, projection_type='natural earth'),
            margin={"r":0,"t":40,"l":0,"b":0},
            coloraxis_colorbar=dict(title=f"{map_cost_metric[1]} (USD)")
        )
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("Note: 'Total Annual Cost', 'Tuition', and 'Monthly Rent' are averaged across all programs in each country. 'Insurance' and 'Visa Fee' are standard costs per country.")


# --- TAB 2: Country Ranker ---
with tab2:
    st.header("Rank Countries by Costs")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    with c1:
        bar_cost_metric = st.selectbox("Select Cost Metric:", options=[("total_annual_cost", "Total Annual Cost"), ("tuition_usd", "Tuition"), ("rent_usd", "Monthly Rent"), ("insurance_usd", "Insurance"), ("visa_fee_usd", "Visa Fee")], format_func=lambda x: x[1], key="bar_cost")
    with c2:
        degree_level = st.selectbox("Select Degree Level:", options=["All"] + df["level"].unique().tolist())
    with c3:
        field_of_study = st.selectbox("Select Field of Study:", options=["All"] + df["field_of_study"].unique().tolist())
    with c4:
        top_n = st.slider("How many countries?", 5, 25, 10, step=1)
        sort_dir = st.radio("Order", ["Highest first", "Lowest first"], horizontal=True)

    # Filter data based on user selections.
    filtered_df = df.copy()
    if degree_level != "All":
        filtered_df = filtered_df[filtered_df["level"] == degree_level]
    if field_of_study != "All":
        filtered_df = filtered_df[filtered_df["field_of_study"] == field_of_study]

    # Use .first() for standard costs, .mean() for variable costs.
    agg_func = 'first' if bar_cost_metric[0] in ['insurance_usd', 'visa_fee_usd'] else 'mean'

    ascending = sort_dir == "Lowest first"
    top_countries = filtered_df.groupby("country")[bar_cost_metric[0]].agg(agg_func).sort_values(ascending=ascending).head(top_n).reset_index()

    fig_bar = px.bar(
        top_countries,
        x="country",
        y=bar_cost_metric[0],
        title=f"Top {top_n} Countries for {bar_cost_metric[1]}",
        labels={"country": "Country", bar_cost_metric[0]: f"Average {bar_cost_metric[1]} (USD)"},
        color=bar_cost_metric[0],
        color_continuous_scale=px.colors.sequential.Plasma,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("Note: Variable costs (Tuition, Rent) are averaged based on the filters. Fixed costs (Insurance, Visa) are standard per country.")

    # Box plot for regional distribution analysis.
    st.subheader("Distribution by Region")
    fig_region = px.box(
        filtered_df,
        x="region",
        y=bar_cost_metric[0],
        points="outliers",
        color="region",
        labels={"region": "Region", bar_cost_metric[0]: f"{bar_cost_metric[1]} (USD)"},
        title=f"Distribution of {bar_cost_metric[1]} by Region",
    )
    fig_region.update_layout(showlegend=False)
    st.plotly_chart(fig_region, use_container_width=True)

# --- TAB 3: Cost Composition ---
with tab3:
    st.header("Cost Composition Breakdown")

    all_countries = sorted(df['country'].unique())
    selected_countries = st.multiselect("Select countries to compare in the chart:", options=all_countries, default=['Germany', 'Austria', 'Turkey'])

    if selected_countries:
        sunburst_df = df[df['country'].isin(selected_countries)].copy()

        # Aggregate and melt data for sunburst visualization.
        cost_breakdown = sunburst_df.groupby('country')[['tuition_usd', 'rent_usd', 'insurance_usd', 'visa_fee_usd']].mean().reset_index()
        sunburst_agg = pd.melt(
            cost_breakdown,
            id_vars=['country'],
            value_vars=['tuition_usd', 'rent_usd', 'insurance_usd', 'visa_fee_usd'],
            var_name='cost_type',
            value_name='average_cost'
        )
        sunburst_agg['cost_type'] = sunburst_agg['cost_type'].replace({'tuition_usd': 'Tuition', 'rent_usd': 'Monthly Rent', 'insurance_usd': 'Insurance', 'visa_fee_usd': 'Visa Fee'})

        fig_sunburst = px.sunburst(
            sunburst_agg,
            path=['country', 'cost_type'],
            values='average_cost',
            title="Comparing Average Cost Composition",
            color='average_cost',
            color_continuous_scale=px.colors.sequential.Plasma
        )
        fig_sunburst.update_traces(hovertemplate='<b>%{label}</b><br>Average Cost: $%{value:,.0f}')
        fig_sunburst.update_layout(height=800, margin=dict(t=40, l=0, r=0, b=0))
        st.plotly_chart(fig_sunburst, use_container_width=True)
    else:
        st.warning("Please select at least one country to display the chart.")
