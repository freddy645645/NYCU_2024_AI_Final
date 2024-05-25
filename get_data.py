import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import os, sys
import argparse

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

import itertools

# Constants
START_DATE = '2023-10-25'
END_DATE = '2024-04-24'
TICKER_LIST = 'DOW_30_TICKER'
FILE_PATH = 'trade_data.csv'

def fetch_data(start_date, end_date, ticker_list):
    try:
        df_raw = YahooDownloader(start_date = start_date,
                         end_date = end_date,
                         ticker_list = ticker_list).fetch_data()
    except Exception as e:
        print("Yahoo data download failed: ", e)
        sys.exit()
    print(df_raw.head())
    return df_raw

def preprocess_data(df_raw):
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list = INDICATORS,
        use_turbulence=False,
        user_defined_feature = False)

    try:
        processed = fe.preprocess_data(df_raw)
    except Exception as e:
        print(e)
        sys.exit()
    
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    processed_full = processed_full.fillna(0)
    print(processed_full.head(5))
    print(processed_full.tail(5))
    return processed_full
    

if __name__ == '__main__':
    os.system('clear' if os.name == 'posix' else 'cls')
    
    # Handle command line arguments
    parser = argparse.ArgumentParser(description='Description of program')
    parser.add_argument('--start_date', default=START_DATE, help='Start date (default: {})'.format(START_DATE))
    parser.add_argument('--end_date', default=END_DATE, help='End date (default: {})'.format(END_DATE))
    parser.add_argument('--train', action='store_true', help='Set data file to train_data.csv with higher priority than --data_file')
    parser.add_argument('--data_file', default=FILE_PATH, help='Data file (default: {})'.format(FILE_PATH))
    parser.add_argument('--ticker_list', default=TICKER_LIST, help='Ticker list in config_tickes.py (default: {})'.format(TICKER_LIST))
    parser.add_argument('--ticker', default=None, help='Single ticker to fetch data for (default: None). If provided, will override ticker_list')
    args = parser.parse_args()
    
    START_DATE = args.start_date
    END_DATE = args.end_date
    FILE_PATH = args.data_file
    if args.ticker:
        TICKER_LIST = args.ticker
        ticker_list = [args.ticker]
    else:
        try:
            ticker_list = getattr(config_tickers, args.ticker_list)
        except AttributeError:
            print(f"Ticker list {args.ticker_list} not found in config_tickers.py")
            sys.exit()
            
    print(f"TRADE_START_DATE: {START_DATE},")
    print(f"TRADE_END_DATE: {END_DATE},")
    print(f"FILE_PATH: {FILE_PATH},")
    print(f"TICKER_LIST: {TICKER_LIST}")
    
    # Fetch data
    df_raw = fetch_data(START_DATE, END_DATE, ticker_list)
    
    # Preprocess data
    data = preprocess_data(df_raw)
    if args.train:
        FILE_PATH = 'train_data.csv'
    data.to_csv(FILE_PATH)
    print("=============================================")
    print(f"Data saved to {FILE_PATH}")
    print(f"[{data.shape[0]} rows x {data.shape[1]} columns]")
    