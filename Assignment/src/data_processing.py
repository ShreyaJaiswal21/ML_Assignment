# src/data_processing.py
import pandas as pd
import json
import os
import logging

# Configuring the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_baanknet_data(file_path):
    """
    Loads and flattens the main baanknet JSON file.
    Handles potential errors during file loading and processing.
    """
    logging.info(f"Loading data from {file_path}...")
    try:
        df = pd.read_json(file_path)
        if 'respData' not in df.columns:
            logging.error("'respData' column not found.")
            return pd.DataFrame()

        data_list = df['respData'].dropna().tolist()
        flat_df = pd.json_normalize(data_list)
        logging.info(f"Successfully loaded and flattened {len(flat_df)} records from baanknet.")
        return flat_df
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return pd.DataFrame()

def load_property_details_data(dir_path):
    """
    Loads, parses, and harmonizes all individual JSON files from the property_details directory.
    """
    logging.info(f"Loading individual property files from {dir_path}...")
    all_properties = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(dir_path, file_name)
            try:
                with open(file_path, 'r') as f:
                    content = json.load(f)
                    if content.get('success') and 'data' in content:
                        data = content['data']
                        # Schema mapping to standardize column names
                        harmonized_data = {
                            'propertyPrice': data.get('reserve_price'),
                            'city': data.get('city'),
                            'propertyType': data.get('name'),
                            'summaryDesc': data.get('address'), # Use address as description
                            'visitOrCount': data.get('emd_amount'), # Example mapping
                            'bankName': data.get('bank_name')
                        }
                        all_properties.append(harmonized_data)
            except Exception as e:
                logging.warning(f"Could not parse {file_name}: {e}")
    
    logging.info(f"Successfully loaded {len(all_properties)} records from property_details directory.")
    return pd.DataFrame(all_properties)

def create_unified_dataset(baanknet_path, details_dir_path):
    """
    Creates a single, unified dataset from the two data sources.
    """
    baanknet_df = load_baanknet_data(baanknet_path)
    details_df = load_property_details_data(details_dir_path)

    # Combining the two dataframes
    final_df = pd.concat([baanknet_df, details_df], ignore_index=True, sort=False)
    
    logging.info(f"Unified dataset created with {len(final_df)} total records.")
    return final_df
