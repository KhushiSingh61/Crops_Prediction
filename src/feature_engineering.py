import pandas as pd
import yaml

with open("configs/params.yaml") as f:
    params = yaml.safe_load(f)

def engineer_features():
    df = pd.read_csv(params["data"]["processed_path"])
    df["Yield_per_Area"] = df["Yield"] / df["Area"]
    df.to_csv(params["data"]["features_path"], index=False)

if __name__ == "__main__":
    engineer_features()
