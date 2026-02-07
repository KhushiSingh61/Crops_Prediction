import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

def evaluate():
    model = mlflow.sklearn.load_model("runs:/latest/model")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    with mlflow.start_run():
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

if __name__ == "__main__":
    evaluate()
