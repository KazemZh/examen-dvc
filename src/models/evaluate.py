import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json

def evaluate_model(input_dir, model_path, metrics_dir, predictions_dir):
    # Load test data
    X_test = pd.read_csv(f"{input_dir}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{input_dir}/y_test.csv")

    # Select only numeric columns
    numeric_columns = X_test.select_dtypes(include=["float64", "int64"]).columns
    X_test = X_test[numeric_columns]

    # Load the trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Make predictions
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    predictions = model.predict(X_test)
    pd.DataFrame(predictions, columns=["prediction"]).to_csv(f"{predictions_dir}/prediction.csv", index=False)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    metrics = {"MSE": mse, "R2": r2}

    # Save metrics
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    with open(f"{metrics_dir}/scores.json", "w") as f:
        json.dump(metrics, f)
    print(f"Evaluation metrics saved to {metrics_dir}/scores.json")

if __name__ == "__main__":
    evaluate_model("data/processed", "models/gbr_model.pkl", "metrics", "data/processed")
