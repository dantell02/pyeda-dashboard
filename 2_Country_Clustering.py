"""
Streamlit page for the 'Interactive Country Clustering' section.
Allows users to dynamically group countries based on selected cost factors
and clustering algorithms.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import pycountry
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from kneed import KneeLocator # Import the elbow detection library
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
        "Could not find the 'preprocessing.py' module. Please ensure it's in the 'src/' directory."
    )
    st.stop()


@st.cache_data
def load_data():
    """Loads and caches the preprocessed education data."""
    return load_and_preprocess_education_data()


df = load_data()

# --- Page Layout ---
st.title("Country Clustering")
st.caption(
    "Group countries based on different cost factors. Follow the sidebar to create clusters."
)

# --- Sidebar Controls ---
st.sidebar.header("Model Configuration")

# Feature Selection widget.
feature_options = {
    "tuition_usd": "Average Tuition",
    "living_cost_index": "Living Cost Index",
    "rent_usd": "Average Rent",
    "visa_fee_usd": "Visa Fee",
    "insurance_usd": "Insurance Cost",
}
selected_features = st.sidebar.multiselect(
    "Features",
    options=list(feature_options.keys()),
    default=["tuition_usd", "rent_usd"],
    format_func=lambda x: feature_options[x],
)

# Model Selection widget.
model_name = st.sidebar.selectbox(
    "Clustering Algorithm", ("K-Means", "Hierarchical Clustering", "DBSCAN")
)

# Model-specific hyperparameter widgets.
if model_name in ["K-Means", "Hierarchical Clustering"]:
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
elif model_name == "DBSCAN":
    eps = st.sidebar.slider(
        "DBSCAN Epsilon (eps)", 0.1, 2.0, 0.5, 0.1,
        help="The maximum distance between samples to be considered neighbors.",
    )

if st.sidebar.button("Create Clusters"):
    if not selected_features:
        st.sidebar.warning("Please select at least one feature.")
    else:
        # Aggregate data into country profiles based on selected features.
        country_profiles = df.groupby("country")[selected_features].mean().reset_index()
        X = country_profiles[selected_features]
        X_scaled = StandardScaler().fit_transform(X)

        # Initialize and run the selected clustering model.
        if model_name == "K-Means":
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
        elif model_name == "Hierarchical Clustering":
            model = AgglomerativeClustering(n_clusters=k)
        elif model_name == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=2)

        country_profiles["cluster"] = model.fit_predict(X_scaled)

        # Store results in session state to persist across reruns.
        st.session_state["final_clusters"] = country_profiles.to_dict()
        st.session_state["final_features"] = selected_features
        st.session_state["model_name"] = model_name
        st.session_state["X_scaled_for_viz"] = X_scaled
        st.session_state["country_labels_for_viz"] = country_profiles["country"].tolist()

# --- Main Panel Display ---

if "final_clusters" in st.session_state:
    st.header(f"Results for {st.session_state['model_name']}")
    final_df = pd.DataFrame(st.session_state["final_clusters"])
    X_scaled_viz = st.session_state["X_scaled_for_viz"]

    # Calculate and display Silhouette Score for cluster evaluation.
    n_clusters = final_df["cluster"].nunique()
    if -1 in final_df["cluster"].unique():
        n_clusters -= 1

    if n_clusters > 1:
        score = silhouette_score(X_scaled_viz, final_df["cluster"])
        st.metric(
            "Silhouette Score",
            f"{score:.3f}",
            help="Measures how well-separated clusters are. Higher is better (max 1.0).",
        )
    else:
        st.info("Silhouette Score requires at least 2 clusters to be calculated.")

    # --- NEW: Optional, expandable Elbow Plot for K-Means ---
    if st.session_state["model_name"] == "K-Means":
        with st.expander("View Elbow Method for Optimal k"):
            st.caption("This plot helps justify the choice of 'k' (number of clusters). The 'elbow' of the curve suggests an optimal value.")

            @st.cache_data
            def calculate_elbow(scaled_data):
                """Calculates inertias and finds the optimal k using the KneeLocator."""
                inertias = []
                k_range = range(1, 11)
                for i in k_range:
                    kmeans_test = KMeans(n_clusters=i, random_state=42, n_init=10)
                    kmeans_test.fit(scaled_data)
                    inertias.append(kmeans_test.inertia_)

                kneedle = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
                elbow_point = kneedle.elbow if kneedle.elbow else 3
                return k_range, inertias, elbow_point

            k_range, inertias, elbow_k = calculate_elbow(X_scaled_viz)

            fig_elbow = px.line(
                x=k_range,
                y=inertias,
                title='Elbow Method for K-Means',
                labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'},
                markers=True
            )

            if len(inertias) >= elbow_k:
                fig_elbow.add_annotation(
                    x=elbow_k, y=inertias[elbow_k-1],
                    text=f"Optimal k = {elbow_k}",
                    showarrow=True, arrowhead=1,
                    ax=-40, ay=-40
                )
            st.plotly_chart(fig_elbow, use_container_width=True)


    # Display model-specific insights (outliers for DBSCAN, dendrogram for Hierarchical).
    if st.session_state["model_name"] == "DBSCAN":
        outliers = final_df[final_df["cluster"] == -1]
        if not outliers.empty:
            st.warning(
                f"**Outliers Detected:** The following countries did not fit into any cluster: {', '.join(outliers['country'].tolist())}"
            )

    if st.session_state["model_name"] == "Hierarchical Clustering":
        st.subheader("Dendrogram")
        st.markdown(
            "This tree diagram shows how the algorithm grouped countries together."
        )
        fig_dendro = ff.create_dendrogram(
            X_scaled_viz,
            orientation="bottom",
            labels=st.session_state["country_labels_for_viz"],
        )
        st.plotly_chart(fig_dendro, use_container_width=True)

    # Create visualization tabs.
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺️ Map View", "📊 Scatter Plot", "📋 Cluster Profiles", "🔎 Cluster Details"]
    )

    with tab1:
        # Map View of cluster results.
        @st.cache_data
        def get_iso_alpha(country_name):
            country_map = {
                "USA": "USA",
                "UK": "GBR",
                "South Korea": "KOR",
                "UAE": "ARE",
                "Turkey": "TUR",
            }
            if country_name in country_map:
                return country_map[country_name]
            try:
                return pycountry.countries.search_fuzzy(country_name)[0].alpha_3
            except LookupError:
                return None

        final_df["iso_alpha"] = final_df["country"].apply(get_iso_alpha)

        fig_map = px.choropleth(
            final_df,
            locations="iso_alpha",
            color="cluster",
            hover_name="country",
            custom_data=["country", "cluster"],
            color_continuous_scale=px.colors.qualitative.Plotly,
            title="Geographic Distribution of Clusters",
        )
        fig_map.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>Cluster: %{customdata[1]}<extra></extra>"
        )
        fig_map.update_layout(height=600)
        st.plotly_chart(fig_map, use_container_width=True)

    with tab2:
        # Scatter plot visualization, dynamically adapting to the number of features.
        num_features = len(st.session_state["final_features"])
        human_readable_labels = {k: v for k, v in feature_options.items()}
        human_readable_labels["pc1"] = "Principal Component 1"
        human_readable_labels["pc2"] = "Principal Component 2"

        if num_features > 2:
            # Use PCA for >2 dimensions.
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled_viz)
            final_df["pc1"] = components[:, 0]
            final_df["pc2"] = components[:, 1]
            fig_scatter = px.scatter(
                final_df,
                x="pc1",
                y="pc2",
                color="cluster",
                hover_name="country",
                title="Cluster Visualization using PCA",
                labels=human_readable_labels,
                custom_data=["country"],
            )
            fig_scatter.update_traces(
                hovertemplate="<b>%{customdata[0]}</b><br>Cluster: %{marker.color}<extra></extra>"
            )
        elif num_features == 2:
            # Use original features for 2 dimensions.
            x_axis, y_axis = st.session_state["final_features"]
            fig_scatter = px.scatter(
                final_df,
                x=x_axis,
                y=y_axis,
                color="cluster",
                title=f"Clusters by {human_readable_labels[x_axis]} vs. {human_readable_labels[y_axis]}",
                labels=human_readable_labels,
                custom_data=["country"],
            )
            fig_scatter.update_traces(
                hovertemplate="<b>%{customdata[0]}</b><br>Cluster: %{marker.color}<extra></extra>"
            )
        elif num_features == 1:
            # Use a strip plot for 1 dimension.
            feature = st.session_state["final_features"][0]
            fig_scatter = px.strip(
                final_df,
                x=feature,
                y="cluster",
                color="cluster",
                orientation="h",
                title=f"Cluster Distribution for {human_readable_labels[feature]}",
                labels=human_readable_labels,
                custom_data=["country"],
            )
            fig_scatter.update_traces(
                hovertemplate="<b>%{customdata[0]}</b><br>Cluster: %{y}<extra></extra>"
            )
        else:
            fig_scatter = None
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        # Table of average feature values for each cluster.
        st.markdown("This table shows the average profile for each generated cluster.")
        summary_df = final_df[final_df["cluster"] != -1]
        cluster_summary = summary_df.groupby("cluster")[
            st.session_state["final_features"]
        ].mean()
        st.dataframe(cluster_summary.style.format("{:,.2f}"))

    with tab4:
        # Detailed breakdown of a user-selected cluster.
        st.markdown(
            "Select a specific cluster to see a detailed breakdown of its members."
        )

        cluster_list = sorted(final_df["cluster"].unique())
        selected_cluster = st.selectbox(
            "Select a cluster to analyze:", options=cluster_list
        )

        if selected_cluster is not None:
            cluster_df = final_df[final_df["cluster"] == selected_cluster]
            st.write(
                f"**Countries in Cluster {selected_cluster}:** {', '.join(cluster_df['country'].tolist())}"
            )

            st.markdown("---")
            st.markdown("**Distribution of Features within this Cluster:**")
            for feature in st.session_state["final_features"]:
                fig_box = px.box(
                    cluster_df,
                    y=feature,
                    title=f"Distribution of '{feature_options[feature]}' for Cluster {selected_cluster}",
                    points="all",
                )
                st.plotly_chart(fig_box, use_container_width=True)
