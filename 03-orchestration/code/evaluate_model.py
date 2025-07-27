import pandas as pd
import pickle
import argparse
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model for NYC taxi duration prediction.')
    parser.add_argument('--year', type=int, required=True, help='Year of the validation data')
    parser.add_argument('--month', type=int, required=True, help='Month of the validation data')
    parser.add_argument('--model-path', type=str, default='models/preprocessor.b', help='Path to the preprocessor/model file')
    args = parser.parse_args()

    val_path = f'data/processed_{args.year}_{args.month:02d}.parquet'
    df_val = pd.read_parquet(val_path)

    with open(args.model_path, "rb") as f_in:
        dv = pickle.load(f_in)

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts_val = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(dicts_val)
    y_val = df_val['duration'].values

    # Dummy model prediction (replace with actual model loading and prediction)
    # y_pred = model.predict(X_val)
    # For now, just use mean as a placeholder
    y_pred = [y_val.mean()] * len(y_val)
    rmse = mean_squared_error(y_val, y_pred)
    print(f"Validation RMSE: {rmse}")
