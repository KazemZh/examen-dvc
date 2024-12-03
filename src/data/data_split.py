import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_path, output_dir):
    # Load the dataset
    data = pd.read_csv(input_path)
    X = data.iloc[:, :-1]  # All columns except the last
    y = data.iloc[:, -1]   # The last column

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the splits
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

if __name__ == "__main__":
    split_data("data/raw/raw.csv", "data/processed")
