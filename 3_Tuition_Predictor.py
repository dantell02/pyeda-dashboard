"""
Streamlit page for the 'Tuition Predictor' section.
Provides an interactive tool to estimate tuition costs using various
regression models and feature configurations.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.express as px
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from numpy import log1p, expm1

# --- Page Configuration ---
st.set_page_config(page_title="Tuition Predictor", page_icon="🧮", layout="wide")
st.title("Tuition Predictor")
st.caption("Estimate tuition based on destination and program characteristics.")

# --- Data Loading ---
# Add parent directory to path to import preprocessing module.
try:
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
    from preprocessing import load_and_preprocess_education_data
except (ModuleNotFoundError, ImportError):
    st.error("Could not import 'load_and_preprocess_education_data' from preprocessing.py.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data():
    """Loads and caches the preprocessed education data."""
    return load_and_preprocess_education_data()

df = load_data()

if df.empty or "tuition_usd" not in df.columns:
    st.error("Preprocessed data is empty or missing 'tuition_usd'.")
    st.stop()

# --- Sidebar - Model & Feature Configuration ---
st.sidebar.header("Model Configuration")

# Define candidate features and populate based on available columns.
num_candidates = ["duration_years", "living_cost_index", "rent_usd", "visa_fee_usd", "insurance_usd"]
cat_candidates = ["country", "level", "field_of_study", "university", "program", "city"]
all_num_features = [c for c in num_candidates if c in df.columns]
all_cat_features = [c for c in cat_candidates if c in df.columns]

# Set sensible default features.
default_num = [f for f in ["duration_years", "living_cost_index"] if f in all_num_features]
default_cat = [f for f in ["country", "level", "field_of_study"] if f in all_cat_features]

num_features = st.sidebar.multiselect("Numeric features", options=all_num_features, default=default_num)
cat_features = st.sidebar.multiselect("Categorical features", options=all_cat_features, default=default_cat)

if not num_features and not cat_features:
    st.sidebar.warning("Select at least one feature.")

# Model and hyperparameter selection widgets.
model_name = st.sidebar.selectbox("Model", ["RandomForestRegressor", "Ridge (linear)", "LinearRegression"], index=0)
n_splits = st.sidebar.slider("Cross-validation folds", min_value=3, max_value=10, value=5)
use_log_target = st.sidebar.checkbox("Use log target (stabilize skew)", value=False, help="Wraps the model with log1p/expm1 transform for tuition.")

if model_name == "RandomForestRegressor":
    n_estimators = st.sidebar.slider("n_estimators", 100, 800, 300, step=50)
    max_depth = st.sidebar.select_slider("max_depth", options=[None, 5, 8, 12, 16, 24, 32], value=None)
else:
    n_estimators = None
    max_depth = None

# --- Train and Evaluate Model ---
st.subheader("Training & Evaluation")

use_cols = num_features + cat_features + ["tuition_usd"]
data = df[use_cols].dropna(subset=["tuition_usd"]).copy()

# Define preprocessing pipelines for numeric and categorical features.
num_pipeline = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipeline = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))])

transformers = []
if num_features:
    transformers.append(("num", num_pipeline, num_features))
if cat_features:
    transformers.append(("cat", cat_pipeline, cat_features))
preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

# Select the base model based on user input.
if model_name == "RandomForestRegressor":
    base_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
elif model_name == "Ridge (linear)":
    base_model = Ridge(alpha=1.0)
else:
    base_model = LinearRegression()

# Build the full pipeline, optionally wrapping with a log-transform for the target.
base_pipe = Pipeline([("pre", preprocessor), ("model", base_model)])
pipe = TransformedTargetRegressor(regressor=base_pipe, func=log1p, inverse_func=expm1) if use_log_target else base_pipe

# Generate cross-validated predictions for robust evaluation.
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
y = data["tuition_usd"].values
X = data.drop(columns=["tuition_usd"])

with st.spinner("Running cross-validation..."):
    y_pred_cv = cross_val_predict(pipe, X, y, cv=kf, n_jobs=-1)

# Calculate and display a suite of evaluation metrics.
rmse = float(np.sqrt(mean_squared_error(y, y_pred_cv)))
mae = float(mean_absolute_error(y, y_pred_cv))
r2 = float(r2_score(y, y_pred_cv))
smape = float(np.mean(2.0 * np.abs(y - y_pred_cv) / (np.abs(y) + np.abs(y_pred_cv) + 1e-9)) * 100)
medae = float(np.median(np.abs(y - y_pred_cv)))

colA, colB, colC, colD, colE = st.columns(5)
colA.metric("RMSE (USD)", f"{rmse:,.0f}", help="Root Mean Squared Error: average size of errors, penalizing large ones more strongly.")
colB.metric("MAE (USD)", f"{mae:,.0f}", help="Mean Absolute Error: average size of prediction errors in USD.")
colC.metric("Median AE (USD)", f"{medae:,.0f}", help="Median Absolute Error: the middle error size; less sensitive to outliers.")
colD.metric("SMAPE (%)", f"{smape:0.1f}%", help="Symmetric MAPE: average percentage error, stable when actuals are near zero.")
colE.metric("R²", f"{r2:0.3f}", help="R-squared: proportion of variance in tuition explained by the model (1.0 = perfect fit).")

# Display diagnostic plots for the cross-validation results.
eval_df = pd.DataFrame({"Actual": y, "Predicted": y_pred_cv})
st.plotly_chart(px.scatter(eval_df, x="Actual", y="Predicted", trendline="ols", title="Predicted vs Actual (Cross-Validation)", labels={"Actual": "Actual Tuition (USD)", "Predicted": "Predicted (USD)"}).update_layout(height=420), use_container_width=True)
st.plotly_chart(px.histogram(x=(y - y_pred_cv), nbins=50, labels={"x": "Residual (Actual − Predicted)"}, title="Residuals (Cross-Validation)").update_layout(height=320), use_container_width=True)

# Fit the final pipeline on all data for the prediction form.
pipe.fit(X, y)

# --- Prediction UI ---
st.subheader("Make a Prediction")

# Populate dropdowns with options from the dataset.
countries = sorted(df["country"].dropna().unique().tolist()) if "country" in df.columns else []
levels = sorted(df["level"].dropna().unique().tolist()) if "level" in df.columns else []
fields = sorted(df["field_of_study"].dropna().unique().tolist()) if "field_of_study" in df.columns else []

def country_defaults(c):
    """Generates default numeric values based on the median for the selected country."""
    d = df[df["country"] == c] if c and "country" in df.columns else df
    return {
        "duration_years": float(d["duration_years"].median()) if "duration_years" in d.columns else 2.0,
        "living_cost_index": float(d["living_cost_index"].median()) if "living_cost_index" in d.columns else 70.0,
        "rent_usd": float(d["rent_usd"].median()) if "rent_usd" in d.columns else 1200.0,
        "visa_fee_usd": float(d["visa_fee_usd"].median()) if "visa_fee_usd" in d.columns else 200.0,
        "insurance_usd": float(d["insurance_usd"].median()) if "insurance_usd" in d.columns else 800.0,
    }

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    selected_country = c1.selectbox("Country", options=countries) if countries else None
    selected_level = c2.selectbox("Level", options=levels) if levels else None
    selected_field = c3.selectbox("Field of Study", options=fields) if fields else None

    defaults = country_defaults(selected_country)
    n1, n2, n3, n4, n5 = st.columns(5)
    val_duration = n1.number_input("Duration (years)", value=defaults["duration_years"], min_value=0.5, step=0.5, format="%.1f")
    val_lci = n2.number_input("Living Cost Index", value=defaults["living_cost_index"], min_value=0.0, step=0.5, format="%.1f", help="Relative living expenses (NY = 100).")
    val_rent = n3.number_input("Rent (USD)", value=defaults["rent_usd"], min_value=0.0, step=50.0, format="%.0f")
    val_visa = n4.number_input("Visa Fee (USD)", value=defaults["visa_fee_usd"], min_value=0.0, step=10.0, format="%.0f")
    val_ins = n5.number_input("Insurance (USD)", value=defaults["insurance_usd"], min_value=0.0, step=10.0, format="%.0f")

    submitted = st.form_submit_button("Predict Tuition")

if submitted:
    # Construct a single-row dataframe from user inputs for prediction.
    row = {f: [val] for f, val in zip(
        ["duration_years", "living_cost_index", "rent_usd", "visa_fee_usd", "insurance_usd"],
        [val_duration, val_lci, val_rent, val_visa, val_ins]
    )}
    row.update({f: [val] for f, val in zip(
        ["country", "level", "field_of_study"],
        [selected_country, selected_level, selected_field]
    )})

    # Ensure only selected features are in the dataframe.
    X_new = pd.DataFrame(row)[num_features + cat_features]

    try:
        y_hat = float(pipe.predict(X_new)[0])

        # Calculate an empirical prediction interval from the cross-validation residuals.
        resid = y - y_pred_cv
        low_q, high_q = np.quantile(resid, [0.025, 0.975])
        lower = max(0.0, y_hat + low_q)
        upper = y_hat + high_q

        st.success(f"**Predicted Tuition:** ${y_hat:,.0f}")
        st.caption(f"Approx. 95% interval (empirical): ${lower:,.0f} – ${upper:,.0f}")
    except Exception as e:
        st.error(f"Could not make a prediction with the current feature selection. {e}")

# --- Model Insights ---
st.subheader("Model Insights")

with st.expander("Feature Importances / Coefficients"):
    try:
        # Access the trained model and preprocessor from the final pipeline.
        trained_base = pipe.regressor_ if isinstance(pipe, TransformedTargetRegressor) else pipe
        base_model = trained_base.named_steps["model"]
        preproc = trained_base.named_steps["pre"]

        # Extract feature names after one-hot encoding.
        feat_names = []
        if num_features:
            feat_names += list(num_features)
        if cat_features and "cat" in preproc.named_transformers_:
            ohe = preproc.named_transformers_["cat"].named_steps["oh"]
            feat_names += ohe.get_feature_names_out(cat_features).tolist()

        # Display feature importances for tree-based models.
        if hasattr(base_model, "feature_importances_"):
            importances = pd.DataFrame({"feature": feat_names, "importance": base_model.feature_importances_}).sort_values("importance", ascending=False).head(20)
            st.dataframe(importances, use_container_width=True)
        # Display coefficients for linear models.
        elif hasattr(base_model, "coef_"):
            coefs = np.array(base_model.coef_).ravel()
            top = pd.DataFrame({"feature": feat_names, "coefficient": coefs}).sort_values("coefficient", key=np.abs, ascending=False).head(20)
            st.dataframe(top, use_container_width=True)
        else:
            st.info("This model does not expose coefficients/feature importances.")
    except Exception as e:
        st.info(f"Could not compute feature importances. {e}")

# --- Group Metrics ---
with st.expander("Group metrics (fairness & segmentation)"):
    try:
        group_options = [c for c in ["level", "field_of_study", "country"] if c in X.columns]
        if group_options:
            group_col = st.selectbox("Group by:", options=group_options, index=0)
            groups = X[group_col].fillna("Unknown")
            rows = []
            for g in pd.Series(groups).unique():
                idx = groups == g
                if idx.sum() >= max(5, n_splits):  # Avoid tiny groups.
                    rmse_g = float(np.sqrt(mean_squared_error(y[idx], y_pred_cv[idx])))
                    mae_g = float(mean_absolute_error(y[idx], y_pred_cv[idx]))
                    r2_g = float(r2_score(y[idx], y_pred_cv[idx])) if idx.sum() > 1 else np.nan
                    rows.append({"group": g, "n": int(idx.sum()), "RMSE": rmse_g, "MAE": mae_g, "R2": r2_g})
            if rows:
                table = pd.DataFrame(rows).sort_values("RMSE")
                st.dataframe(table, use_container_width=True)
            else:
                st.write("Group sizes are too small for reliable metrics.")
        else:
            st.write("No suitable grouping columns available in the selected features.")
    except Exception as e:
        st.info(f"Could not compute group metrics. {e}")
