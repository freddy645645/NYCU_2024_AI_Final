import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import os, sys
import argparse

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl import config_tickers
from finrl.config import INDICATORS, DATA_SAVE_DIR

import itertools

# Constants
START_DATE = '2020-07-01'
END_DATE = '2023-12-31'
TICKER_LIST = 'DOW_30_TICKER'
FILE_PATH = 'trade_data.csv'

def handle_args():
    parser = argparse.ArgumentParser(
        description='Fetch and preprocess data for training or testing.',
        usage='python get_data.py [--train] [-s START_DATE] [-e END_DATE] [-p FILE_PATH] [-T TICKER_LIST] [-t TICKER [TICKER ...]]',
        epilog='Example: python get_data.py -s 2023-10-25 -e 2024-04-24 -p trade_data.csv -T DOW_30_TICKER'
    )
    
    parser.add_argument('-i', '--interactive',
                        action='store_true',
                        help='Interactive mode')
                                     
    parser.add_argument('--train', 
                        action='store_true',
                        help='Set data file to train_data.csv')
    parser.add_argument('-s', '--start_date', 
                        default=START_DATE,
                        type=str,
                        metavar='',
                        help='Start date (default: {})'.format(START_DATE))
    parser.add_argument('-e','--end_date',
                        default=END_DATE, 
                        metavar='', 
                        type=str, 
                        help='End date (default: {})'.format(END_DATE))
    parser.add_argument('-p','--file_path', 
                        default=FILE_PATH,
                        metavar='',
                        type=str,  
                        help='Data file (default: {})'.format(FILE_PATH))
    parser.add_argument('-T','--ticker_list', 
                        default=TICKER_LIST, 
                        metavar='',
                        type=str,
                        help='Ticker list in config_tickes.py (default: {})'.format(TICKER_LIST))
    parser.add_argument('-t','--tickers', 
                        default=None,
                        type=str, 
                        nargs='+',
                        help='''Specific tickers to fetch data for (default: None). 
                                If provided, will override ticker_list.''')
    args = parser.parse_args()
    return args

def interact():
    print("Interactive mode: Enter data parameters")
    start_date = str(input(f"Start date ({START_DATE}): ")) or START_DATE
    end_date = str(input(f"End date ({END_DATE}): ")) or END_DATE
    file_path = str(input(f"Data file ({FILE_PATH}): ")) or FILE_PATH
    print(f"\nTicker list in config_tickers.py. Enter 'help' for list.")
    ticker_list = str(input(f"Ticker list in config_tickers.py ({TICKER_LIST}): ")) or TICKER_LIST
    if ticker_list == 'help':
        print("\nAvailable ticker lists in config_tickers.py:")
        print([attr for attr in dir(config_tickers) if not attr.startswith('__')])
        ticker_list = str(input(f"Ticker list in config_tickers.py ({TICKER_LIST}): ")) or TICKER_LIST
    print("\nIf you want to fetch data for specific tickers, enter them below. It will override the ticker list.")
    tickers = input("Tickers (separated by space, default: None): ").split()
    return argparse.Namespace(interactive=True, train=False, start_date=start_date, end_date=end_date, file_path=file_path, ticker_list=ticker_list, tickers=tickers)

class DataFetcher:
    def __init__(self, start_date=START_DATE, end_date=END_DATE, ticker_list=TICKER_LIST):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
    
    def __str__(self):
        return f"GetData(start_date={self.start_date}, end_date={self.end_date}, ticker_list={self.ticker_list})"
    
    def fetch_data(self):
        try:
            df_raw = YahooDownloader(start_date = self.start_date,
                            end_date = self.end_date,
                            ticker_list = self.ticker_list).fetch_data()
        except Exception as e:
            print("Yahoo data download failed: ", e)
            sys.exit()
        print(df_raw.head())
        return df_raw
        
    def preprocess_data(self, df_raw):
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
        
    def get_data(self):
        df_raw = self.fetch_data()
        data = self.preprocess_data(df_raw)
        return data
    

if __name__ == '__main__':
    os.system('clear' if os.name == 'posix' else 'cls')
    
    # Handle command line arguments
    """
    
    """
    args = handle_args()
    if args.interactive:
        args = interact()
    START_DATE = args.start_date
    END_DATE = args.end_date
    FILE_PATH = args.file_path
    if args.tickers:
        ticker_list = args.tickers
    else:
        try:
            ticker_list = getattr(config_tickers, args.ticker_list)
        except AttributeError:
            print(f"Ticker list {args.ticker_list} not found in config_tickers.py")
            sys.exit()
    
    # Fetch data
    df = DataFetcher(START_DATE, END_DATE, ticker_list)
    print(df.__str__() + "\n")
    data = df.get_data()
    
    if args.train:
        FILE_PATH = 'train_data.csv'
    if not os.path.exists(DATA_SAVE_DIR):
        os.makedirs(DATA_SAVE_DIR)
    data.to_csv(os.path.join(DATA_SAVE_DIR, FILE_PATH))
    print("=============================================")
    print("Data saved to", os.path.join(DATA_SAVE_DIR, FILE_PATH))
    print(f"[{data.shape[0]} rows x {data.shape[1]} columns]")
    