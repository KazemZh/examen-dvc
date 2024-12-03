import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json

def evaluate_model(input_dir, model_path, output_dir):
    # Load test data
    X_test = pd.read_csv(f"{input_dir}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{input_dir}/y_test.csv")
    
    # Load the trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Make predictions
    predictions = model.predict(X_test)
    pd.DataFrame(predictions, columns=["prediction"]).to_csv(f"{output_dir}/prediction.csv", index=False)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    metrics = {"MSE": mse, "R2": r2}
    
    # Save metrics
    with open(f"{output_dir}/scores.json", "w") as f:
        json.dump(metrics, f)
    print(f"Evaluation metrics saved to {output_dir}/scores.json")

if __name__ == "__main__":
    evaluate_model("data/processed", "models/gbr_model.pkl", "metrics")

