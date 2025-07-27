import pandas as pd
import xgboost as xgb
import pickle
from pathlib import Path
import argparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def train(year, month):
    train_path = f'data/processed_{year}_{month:02d}.parquet'
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    val_path = f'data/processed_{next_year}_{next_month:02d}.parquet'

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts_train = df_train[categorical + numerical].to_dict(orient='records')
    dicts_val = df_val[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(dicts_train)
    X_val = dv.transform(dicts_val)

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    with mlflow.start_run() as run:
        train_dm = xgb.DMatrix(X_train, label=y_train)
        val_dm = xgb.DMatrix(X_val, label=y_val)
        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }
        mlflow.log_params(best_params)
        booster = xgb.train(
            params=best_params,
            dtrain=train_dm,
            num_boost_round=30,
            evals=[(val_dm, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(val_dm)
        rmse = mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
        print(f"MLflow run_id: {run.info.run_id}")
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model for NYC taxi duration prediction.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()
    train(args.year, args.month)
