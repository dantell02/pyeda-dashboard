"""
Main entry point and landing page for the Streamlit multi-page application.
This script sets the main page configuration, serves as the landing page,
and includes a section with details about the dataset.
"""
import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Education Cost Dashboard",
    page_icon="🎓",
    layout="wide"
)

# --- Main Landing Page Content ---
st.title("International Education Cost Dashboard 🎓")
st.markdown("Analyze and compare the costs of university programs across the globe.")
st.sidebar.success("Select a page above to begin.")

# --- Original Welcome Message ---
st.header("Welcome!")
st.write("This dashboard is designed to help you explore the costs of international education.")
st.write("Use the menu on the left to navigate between the different analysis pages:")
st.markdown("""
- **Global Cost Overview**: Explore interactive maps and charts of education costs.
- **Country Clustering**: Group countries by their cost profiles using machine learning.
- **Tuition Predictor**: Get an estimated tuition cost based on your choices.
""")


# --- "About the Data" Section ---
st.divider()
st.header("About the Dataset")

# --- Data Loading ---
DATA_FILENAME = "international_education_costs.csv"

def find_data_file(filename: str) -> Path:
    """Robustly finds the data file by searching parent directories."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        p = parent / "data" / filename
        if p.exists():
            return p
    p = Path.cwd() / "data" / filename
    if p.exists():
        return p
    raise FileNotFoundError(f"Could not find data/{filename}")

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """Loads the raw CSV data, cached for performance."""
    return pd.read_csv(path, low_memory=False)

data_path = find_data_file(DATA_FILENAME)
df = load_data(data_path)

# --- Restored Data Description ---
st.write(
    """
This dataset compiles comparative international education cost information across countries,
cities, and universities. It standardizes key fields—such as tuition in USD, living-cost index,
rent, visa fees, insurance, and exchange rate—to support consistent comparisons of programs and
locations. It is provided as a single CSV on Kaggle with concise, up-to-date fields intended for
exploratory analysis and budgeting use cases. ¹
"""
)


# --- Quick Facts ---
COUNTRY_COL = next((c for c in ["Country", "country", "COUNTRY"] if c in df.columns), None)
last_updated = datetime.fromtimestamp(data_path.stat().st_mtime)

if COUNTRY_COL:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Countries", f"{df[COUNTRY_COL].nunique():,}")
    c4.metric("Last Updated", last_updated.strftime("%Y-%m-%d"))
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Last Updated", last_updated.strftime("%Y-%m-%d"))

# --- Data Dictionary ---
st.subheader("Data Dictionary")
def example_value(s: pd.Series) -> str:
    """Extracts the first non-null value from a Series as an example."""
    ex = s.dropna().head(1)
    return "—" if ex.empty else str(ex.iloc[0])[:80]

DESCRIPTIONS = {
    "Country": "Country where the university is located.",
    "City": "City of the institution.",
    "University": "Official university name.",
    "Program": "Name of the academic program.",
    "Level": "Degree level (e.g., Bachelor, Master).",
    "Duration_Years": "Program duration in years.",
    "Tuition_USD": "Annual tuition in USD.",
    "Living_Cost_Index": "Index approximating living costs.",
    "Rent_USD": "Estimated monthly rent in USD.",
    "Visa_Fee_USD": "Estimated visa fee in USD.",
    "Insurance_USD": "Estimated health insurance cost in USD.",
    "Exchange_Rate": "Exchange rate used to convert local currency to USD (if applicable)."
}
# --- Reordered Columns in Schema ---
schema = pd.DataFrame({
    "Column": df.columns,
    "Type": [str(t) for t in df.dtypes],
    "Example": [example_value(df[c]) for c in df.columns],
    "Description": [DESCRIPTIONS.get(c, "") for c in df.columns]
})
st.dataframe(schema, hide_index=True, use_container_width=True)

# --- Browse the data ---
st.subheader("Browse the Raw Data")

with st.expander("Column visibility & quick search", expanded=True):
    cols_to_show = st.multiselect(
        "Columns to show", options=list(df.columns), default=list(df.columns)
    )
    query = st.text_input("Quick search (matches any visible column)", "")

view = df[cols_to_show] if cols_to_show else df
if query:
    mask = view.astype(str).apply(
        lambda col: col.str.contains(query, case=False, na=False)
    )
    view = view[mask.any(axis=1)]

def fmt_currency_cols(frame: pd.DataFrame) -> pd.DataFrame:
    """Formats columns with names suggesting currency as USD strings."""
    out = frame.copy()
    moneyish = [c for c in out.columns if out[c].dtype.kind in "if" and any(k in c.lower() for k in ["usd", "rent", "fee", "cost", "price"])]
    for c in moneyish:
        out[c] = out[c].map(lambda x: f"${x:,.0f}" if pd.notna(x) else x)
    return out

st.dataframe(fmt_currency_cols(view), use_container_width=True, height=480)

# --- Download Button ---
st.download_button(
    "Download CSV",
    df.to_csv(index=False).encode("utf-8"),
    file_name=DATA_FILENAME,
    mime="text/csv",
)

# --- Citation ---
st.markdown("**Reference (APA):**")
st.markdown(
    """
Shamim, A. (2023). *Cost of International Education* [Data set]. Kaggle.
Retrieved August 26, 2025, from https://www.kaggle.com/datasets/adilshamim8/cost-of-international-education
"""
)
