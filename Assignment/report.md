Report - Real Estate Price Prediction
1. Introduction
This report details the end-to-end process of building a machine learning pipeline to predict real estate prices. The assignment involved significant data engineering challenges due to heterogeneous data sources and messy, unstructured data.

Key Finding: The primary conclusion of this project is that the provided dataset, with only ~80-100 usable records after cleaning, is insufficient to train a reliable predictive model. Both simple and complex models exhibited poor performance (negative R-squared), a clear indication that the models could not generalize from the limited data. The most critical recommendation is the acquisition of a larger, more comprehensive dataset.

2. Data Engineering Pipeline
Data Ingestion and Schema Mapping
Two Sources: Data was ingested from baanknet_property_details.json (a single file with nested objects) and the property_details/ directory (thousands of individual files).

Heterogeneous Schema: The sources had different structures and field names (e.g., propertyPrice vs. reserve_price).

Solution: A Python script (src/data_processing.py) was developed to:

Parse both sources independently.

Flatten the nested baanknet data using pandas.json_normalize.

Harmonize the schema by mapping disparate field names to a unified standard.

Combine the two sources into a single Pandas DataFrame.

    NOTE - An additional python notebook file named ModelTrainColab.ipynb has been attached, containing the model ( namely ridge_model.pkl and scaler.pkl int the folder named Models) that I trained on Google colab.

3. Feature Engineering & NLP
Challenge: Critical features like property area were not available in structured fields.

Solution (Rule-based NLP): A rule-based information extraction system was implemented using regular expressions in src/feature_engineering.py. The regex pattern (\d+\.?\d*)\s*(?:sq\.?ft\.?|sft) was used to extract area_sqft from unstructured text fields like summaryDesc.

Cleaning: Categorical features like city were standardized (e.g., stripping whitespace, title casing).

4. Predictive Modeling
Experiment Tracking with MLflow
All experiments were tracked using MLflow to ensure reproducibility. For each run, we logged parameters, evaluation metrics, and the trained model artifact.

Model Performance and Evaluation
Baseline: A simple Ridge Regression model was tested.

Advanced Model: A RandomForestRegressor was also trained.

Results: Both models performed poorly on the unseen test set, yielding a negative R-squared score of approximately -0.12.

Analysis: A negative R² score signifies that the model's predictions are worse than a naive baseline of simply predicting the mean price. This is a classic symptom of severe overfitting. The model, given too many features (after encoding) and too few data points, memorized the training data and failed to learn generalizable patterns.

Model Interpretability (SHAP)
Despite the poor performance, a SHAP summary plot was generated to understand what features the model attempted to use. The plot showed that the model tried to latch onto features like visitOrCount and specific cities, but these relationships did not hold up on the test data.

(Here, you would embed your SHAP plot image)

5. Model Deployment
REST API: A REST API was built using FastAPI (src/app.py). It exposes a /predict endpoint that accepts property features as a JSON payload and returns the predicted price.

Model Loading: The API loads the latest model directly from the MLflow tracking server, ensuring the most recently validated model is served.

Error Handling: Robust input validation (via Pydantic) and error handling were implemented.

6. System Design for Scalability
To handle a hypothetical load of 1 million listings/day and thousands of requests/minute, the following architecture is proposed:

Ingestion: An Apache Kafka message queue to handle high-throughput data streams.

Processing: Spark Streaming to consume from Kafka, performing real-time cleaning and feature extraction.

Storage: A Data Lake (S3) for raw data and a Data Warehouse (Snowflake/BigQuery) for structured data used in training.

Training: An Apache Airflow DAG to orchestrate daily batch retraining jobs, with models versioned in MLflow.

Serving: The model API containerized with Docker and deployed on Kubernetes for auto-scaling and high availability.

7. Conclusion and Recommendations
The project successfully demonstrates an end-to-end MLOps pipeline, from data ingestion to deployment. However, the most significant outcome is the data-driven conclusion that a reliable prediction model is not feasible with the current dataset.

Primary Recommendation: The top priority must be data acquisition. A dataset with thousands of clean, well-structured records is required to train a model that can provide meaningful business value.
