import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_data(input_dir, output_dir):
    # Load the datasets
    X_train = pd.read_csv(f"{input_dir}/X_train.csv")
    X_test = pd.read_csv(f"{input_dir}/X_test.csv")
    
    # Normalize the datasets
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the normalized datasets
    pd.DataFrame(X_train_scaled).to_csv(f"{output_dir}/X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled).to_csv(f"{output_dir}/X_test_scaled.csv", index=False)

if __name__ == "__main__":
    normalize_data("data/processed", "data/processed")

