import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import os

def grid_search(input_dir, output_path):
    # Load training data
    X_train = pd.read_csv(f"{input_dir}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{input_dir}/y_train.csv")

    # Select only numeric columns
    numeric_columns = X_train.select_dtypes(include=["float64", "int64"]).columns
    X_train = X_train[numeric_columns]

    # Define the model and hyperparameters
    model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    }

    # Perform GridSearch
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring="r2", verbose=2)
    grid_search.fit(X_train, y_train.values.ravel())

    # Save the best parameters
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, "wb") as f:
        pickle.dump(grid_search.best_params_, f)
    print(f"Best parameters saved to {output_path}")

if __name__ == "__main__":
    grid_search("data/processed", "models/best_params.pkl")
