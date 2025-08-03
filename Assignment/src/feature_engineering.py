import re
import pandas as pd

def extract_area_from_text(text):
    """
    Extracts area in sq. ft. from a text description using regular expressions.
    This is a rule-based approach for information extraction.
    """
    if not isinstance(text, str):
        return None
    # Regex to find a number (integer or float) followed by variations of "sq.ft."
    match = re.search(r'(\d+\.?\d*)\s*(?:sq\.?ft\.?|sft)', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering steps to the dataframe.
    """
    # Create a new feature 'area_sqft' from the summary description
    if 'summaryDesc' in df.columns:
        df['area_sqft'] = df['summaryDesc'].apply(extract_area_from_text)
    
    # Example: Clean the city column
    if 'city' in df.columns:
        df['city'] = df['city'].str.strip().str.title()

    return df
