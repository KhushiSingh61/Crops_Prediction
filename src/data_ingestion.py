import pandas as pd
import yaml
import os

with open("configs/params.yaml") as f:
    params = yaml.safe_load(f)

def ingest_data():
    os.makedirs("data/raw", exist_ok=True)
    df = pd.read_csv(params["data"]["raw_path"])
    df.to_csv(params["data"]["raw_path"], index=False)

if __name__ == "__main__":
    ingest_data()
