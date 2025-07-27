import pandas as pd
import argparse
from pathlib import Path

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    df.to_parquet(f'data/raw_{year}_{month:02d}.parquet')
    print(f"Downloaded and saved: data/raw_{year}_{month:02d}.parquet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download NYC taxi data for a given year and month.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to download')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to download')
    args = parser.parse_args()
    Path('data').mkdir(exist_ok=True)
    read_dataframe(args.year, args.month)
