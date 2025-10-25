"""
Data loading and preprocessing pipeline for the International Education Costs dataset.
"""
import pandas as pd
import re
from pathlib import Path

def load_education_data():
    """
    Loads the raw International Education Costs dataset from the data directory.
    """
    # Construct a robust path to the data file relative to this script.
    SRC_DIR = Path(__file__).parent
    PROJECT_ROOT = SRC_DIR.parent
    DATA_DIR = PROJECT_ROOT / "data"
    SRC_FILE = DATA_DIR / "international_education_costs.csv"

    df = pd.read_csv(SRC_FILE, encoding='ISO-8859-1')
    return df


def load_and_preprocess_education_data():
    """
    Loads and performs all preprocessing on the education costs dataset.

    This includes:
    - Standardizing column names.
    - Engineering new features ('field_of_study', 'total_annual_cost').
    - Handling outliers using the IQR method.
    - Filtering out unclassified program categories.

    Returns:
        pd.DataFrame: A cleaned and feature-rich dataframe ready for analysis.
    """
    df = load_education_data()

    # Standardize all column names to snake_case for consistency.
    def to_snake(s):
        s = re.sub(r"[^0-9a-zA-Z]+", "_", s.strip())
        s = re.sub(r"_+", "_", s)
        return s.strip("_").lower()
    df.columns = [to_snake(c) for c in df.columns]

    # Drop the exchange_rate column as all monetary values are in USD.
    if 'exchange_rate' in df.columns:
        df = df.drop(columns=['exchange_rate'])

    # Engineer a 'field_of_study' feature by mapping raw program names to broader categories.
    def map_program_to_field(program_name):
        """Categorizes a program name into a broader field of study using keywords."""
        program_name = str(program_name).lower()

        # Keyword lists for each category, ordered from most specific to most general.
        data_ai_keywords = ['data', 'artificial intelligence', 'ai', 'analytics', 'machine learning', 'bioinformatics']
        software_systems_keywords = ['software', 'robotics', 'systems', 'networks', 'cybersecurity', 'information security', 'game technology']
        eng_keywords = ['engineering', 'mechatronics', 'aerospace', 'electronics', 'sustainable energy', 'photonics', 'materials science', 'chemical engineering', 'petroleum engineering']
        business_keywords = ['business', 'management', 'finance', 'mba', 'economics']
        science_keywords = ['physics', 'mathematics', 'chemistry', 'biology', 'environmental', 'biotechnology', 'neuroscience', 'pharmaceutical', 'climate science', 'forestry']
        medicine_keywords = ['medicine', 'medical', 'biomedical']
        design_keywords = ['design', 'architecture', 'media']
        social_keywords = ['social sciences', 'psychology', 'international relations', 'political science']

        if any(keyword in program_name for keyword in data_ai_keywords):
            return 'Data Science & AI'
        elif any(keyword in program_name for keyword in software_systems_keywords):
            return 'Software & Systems'
        elif any(keyword in program_name for keyword in ['computer science', 'information technology', 'computing']):
             return 'Computer Science'
        elif any(keyword in program_name for keyword in medicine_keywords):
            return 'Medicine & Health'
        elif any(keyword in program_name for keyword in eng_keywords):
            return 'Engineering'
        elif any(keyword in program_name for keyword in business_keywords):
            return 'Business & Economics'
        elif any(keyword in program_name for keyword in science_keywords):
            return 'Natural Sciences'
        elif any(keyword in program_name for keyword in design_keywords):
            return 'Arts & Design'
        elif any(keyword in program_name for keyword in social_keywords):
            return 'Social Sciences & Humanities'
        else:
            return 'Other'

    df['field_of_study'] = df['program'].apply(map_program_to_field)

    # --- CORRECTED: Engineer an 'annual_rent_usd' feature ---
    df['annual_rent_usd'] = df['rent_usd'] * 12

    # Engineer a 'total_annual_cost' feature using the new annualized rent.
    df['total_annual_cost'] = df['tuition_usd'] + df['annual_rent_usd'] + df['insurance_usd']

    # Remove outliers from key cost columns using the robust IQR method.
    for col in ['tuition_usd', 'total_annual_cost']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # Filter out the small 'Other' category to ensure clean data for modeling.
    df = df[df['field_of_study'] != 'Other'].copy()

    return df


if __name__ == "__main__":
    # This block allows the script to be run directly for testing.
    processed_df = load_and_preprocess_education_data()
    print("Preprocessing complete.")
    print(f"Data shape after processing: {processed_df.shape}")
    print(processed_df.head())
