import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import joblib # For saving/loading the trained model
from file_reader import load_data_from_db

MODEL_FILE = '../data/patient_risk_model.joblib'

df = load_data_from_db()

if df.empty:
    print("Cannot train model: Database is empty.")

    # Define features (X) and target (y)
features = ['Age', 'Gender', 'Condition', 'Procedure', 'Length_of_Stay']
target = 'Readmission'

    # Drop rows with NaN in features or target, if any (robustness)
df_cleaned = df.dropna(subset=features + [target])
if df_cleaned.empty:
    print("Cannot train model: No complete data rows after dropping NaNs.")

X = df_cleaned[features]
y = df_cleaned[target]

    # Identify categorical and numerical features
categorical_features = ['Gender', 'Condition', 'Procedure']
numerical_features = ['Age', 'Length_of_Stay']

    # Create a column transformer for one-hot encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ],
    remainder='drop' # Drop columns not specified
)

    # Create a pipeline with preprocessing and a Logistic Regression model
model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(solver='liblinear', random_state=42))])

    # Train the model (using all available data for the final model,
    # or you could use train_test_split for evaluation during development)
model.fit(X, y)

    # Save the trained model
joblib.dump(model, MODEL_FILE)
print(f"Model trained and saved to {MODEL_FILE}")