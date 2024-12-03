import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pickle

def grid_search(input_dir, output_path):
    # Load training data
    X_train = pd.read_csv(f"{input_dir}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{input_dir}/y_train.csv")
    
    # Initialize model and parameters
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
    with open(output_path, "wb") as f:
        pickle.dump(grid_search.best_params_, f)
    print(f"Best parameters saved to {output_path}")

if __name__ == "__main__":
    grid_search("data/processed", "models/best_params.pkl")

