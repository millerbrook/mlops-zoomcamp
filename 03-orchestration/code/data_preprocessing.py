import pandas as pd
import argparse
from pathlib import Path

def preprocess(year, month):
    input_path = f'data/raw_{year}_{month:02d}.parquet'
    output_path = f'data/processed_{year}_{month:02d}.parquet'
    df = pd.read_parquet(input_path)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    df.to_parquet(output_path)
    print(f"Preprocessed and saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess NYC taxi data for a given year and month.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to preprocess')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to preprocess')
    args = parser.parse_args()
    Path('data').mkdir(exist_ok=True)
    preprocess(args.year, args.month)
