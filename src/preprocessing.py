import pandas as pd
import yaml
import os

with open("configs/params.yaml") as f:
    params = yaml.safe_load(f)

def preprocess():
    df = pd.read_csv(params["data"]["raw_path"])
    df = df.dropna()
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(params["data"]["processed_path"], index=False)

if __name__ == "__main__":
    preprocess()
