import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

with open("configs/params.yaml") as f:
    params = yaml.safe_load(f)

def train():
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    df = pd.read_csv(params["data"]["features_path"])
    X = df.drop("Yield", axis=1)
    y = df["Yield"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"]
    )

    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=params["model"]["n_estimators"],
            max_depth=params["model"]["max_depth"],
            random_state=params["model"]["random_state"]
        )

        model.fit(X_train, y_train)

        mlflow.log_params(params["model"])
        mlflow.sklearn.log_model(model, "model")

        X_test.to_csv("data/processed/X_test.csv", index=False)
        y_test.to_csv("data/processed/y_test.csv", index=False)

if __name__ == "__main__":
    train()
