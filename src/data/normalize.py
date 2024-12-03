import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def normalize_data(input_dir, output_dir):
    # Load the datasets
    X_train = pd.read_csv(f"{input_dir}/X_train.csv")
    X_test = pd.read_csv(f"{input_dir}/X_test.csv")

    # Select only numeric columns
    numeric_columns = X_train.select_dtypes(include=["float64", "int64"]).columns
    X_train_numeric = X_train[numeric_columns]
    X_test_numeric = X_test[numeric_columns]

    # Normalize the numeric datasets
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)

    # Replace numeric columns with normalized data
    X_train[numeric_columns] = X_train_scaled
    X_test[numeric_columns] = X_test_scaled

    # Save the normalized datasets
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    X_train.to_csv(f"{output_dir}/X_train_scaled.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test_scaled.csv", index=False)

if __name__ == "__main__":
    normalize_data("data/processed", "data/processed")
