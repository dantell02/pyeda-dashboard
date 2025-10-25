# International Education Cost Dashboard

This project is an interactive web application built with Streamlit for analyzing the costs of international education across various countries. It allows users to explore cost data through different visualizations and apply machine learning models to uncover insights.


## Features

-   **Global Cost Overview**: A multi-tab view for exploring costs.
    -   **Interactive World Map**: Visualize average costs per country.
    -   **Dynamic Country Ranker**: Create custom Top-N lists of the most or least expensive countries based on filters for cost type, degree level, and field of study.
    -   **Cost Composition Chart**: An interactive sunburst plot to compare the breakdown of average student costs (tuition, rent, etc.) for a selection of countries.

-   **Country Clustering**: An interactive tool to group countries based on their cost profiles.
    -   **Custom Model Building**: Users can select which cost factors to include and choose from different clustering algorithms.
    -   **Model Evaluation**: Includes the Elbow Method to justify the choice of k for K-Means, a Silhouette Score to measure cluster quality, and model-specific visualizations like dendrograms (for Hierarchical) and outlier detection (for DBSCAN).
    -   **Visualization**: Uses Principal Component Analysis to create the most accurate 2D representation of high-dimensional clusters.

-   **Tuition Predictor**: A comprehensive regression toolkit for estimating tuition costs.
    -   **Interactive Prediction Form**: Users can input program characteristics (country, level, field of study) to get a tuition estimate.
    -   **Model Playground**: Allows selection of different regression models, feature sets, and cross-validation folds.
    -   **In-Depth Evaluation**: Provides a full set of performance metrics (RMSE, R², etc.) and diagnostic plots to assess model accuracy and fairness.


## Project Structure

pyeda-dashboard/
├── src/
│   ├── pages/
│   │   ├── 1_Global_Cost_Overview.py
│   │   ├── 2_Country_Clustering.py
│   │   └── 3_Tuition_Predictor.py
│   ├── Home.py
│   └── preprocessing.py
├── data/
│   └── international_education_costs.csv
├── notebooks/
│   └── EDA.ipynb
└── requirements.txt


## Setup and Installation

1.  Create and activate a virtual environment.
2.  Install the required packages:
    $ pip install -r requirements.txt


## How to Run

Navigate to the project's root directory in your terminal and run the following command:
$ streamlit run src/Home.py
