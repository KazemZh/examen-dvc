import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import os

def train_model(input_dir, params_path, output_path):
    # Load training data
    X_train = pd.read_csv(f"{input_dir}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{input_dir}/y_train.csv")

    # Select only numeric columns
    numeric_columns = X_train.select_dtypes(include=["float64", "int64"]).columns
    X_train = X_train[numeric_columns]

    # Load best parameters
    with open(params_path, "rb") as f:
        best_params = pickle.load(f)

    # Train the model using the best parameters
    model = GradientBoostingRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # Save the trained model
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model trained and saved to {output_path}")

if __name__ == "__main__":
    train_model("data/processed", "models/best_params.pkl", "models/gbr_model.pkl")
