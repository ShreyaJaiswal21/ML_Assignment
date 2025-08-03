import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.data_processing import create_unified_dataset
from src.src.feature_engineering import engineer_features

def train_model():
    """
    Main function to run the model training and evaluation process.
    """
    # 1. Load and process data
    df = create_unified_dataset('data/baanknet_property_details.json', 'data/property_details/')
    df = engineer_features(df)

    # 2. Select features and define target
    # Based on our findings, we use a minimal feature set to avoid overfitting
    feature_columns = ['propertyType', 'city', 'visitOrCount']
    target_column = 'propertyPrice'
    
    # Drop rows where target or key features are missing
    df_model = df[feature_columns + [target_column]].dropna()

    if df_model.empty:
        print("DataFrame is empty after dropping NaNs. Cannot train model.")
        return

    X = df_model[feature_columns]
    y = df_model[target_column]

    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # 3. Start MLflow experiment
    mlflow.set_experiment("Real Estate Price Prediction")

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("features", feature_columns)
        mlflow.log_param("model_type", "RandomForestRegressor")

        # 4. Create preprocessing and modeling pipeline
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        # 5. Split data and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training the model pipeline...")
        model_pipeline.fit(X_train, y_train)
        print("Training complete.")

        # 6. Evaluate and log metrics
        y_pred = model_pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae}, R2: {r2}")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # 7. Log the model
        mlflow.sklearn.log_model(model_pipeline, "price_prediction_model")
        print(f"Model saved. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_model()
