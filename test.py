import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from train import INDICATORS

from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import PPO
from finrl.main import check_and_make_directories
from finrl.config import TRAINED_MODEL_DIR, RESULTS_DIR, DATA_SAVE_DIR
# from finrl.plot import backtest_stats

# Contestants are welcome to split the data in their own way for model tuning
TRAIN_FILE_PATH =  os.path.join(DATA_SAVE_DIR, 'train_data.csv')
TRADE_FILE_PATH = os.path.join(DATA_SAVE_DIR, 'trade_data.csv')
TRADE_START_DATE = pd.read_csv(TRADE_FILE_PATH)['date'].min()
TRADE_END_DATE = pd.read_csv(TRADE_FILE_PATH)['date'].max()

# PPO configs
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0003,
    "batch_size": 128,
}

def handle_diff(trade_file=TRADE_FILE_PATH, train_file=TRAIN_FILE_PATH):
    # Handle the difference between the two datasets
    data1 = pd.read_csv(trade_file)
    data2 = pd.read_csv(train_file)
    
    dim1 = data1.tic.unique()
    dim2 = data2.tic.unique()
    
    diff = set(dim1) - set(dim2)
    for d in diff:
        data1 = data1[data1.tic != d]
        
    return data1


if __name__ == '__main__':
    # We will use unseen, post-deadline data for testing
    parser = argparse.ArgumentParser(description='Description of program')
    parser.add_argument('-s','--start_date', default=TRADE_START_DATE, help='Trade start date (default: {})'.format(TRADE_START_DATE))
    parser.add_argument('-e','--end_date', default=TRADE_END_DATE, help='Trade end date (default: {})'.format(TRADE_END_DATE))
    parser.add_argument('-p','--trade_file_path', default=TRADE_FILE_PATH, help='Trade data file')
    parser.add_argument('-P','--train_file_path', default=TRAIN_FILE_PATH, help='Train data file')    

    args = parser.parse_args()
    TRADE_START_DATE = args.start_date
    TRADE_END_DATE = args.end_date
    
    processed_full = handle_diff(args.trade_file_path, args.train_file_path)
    trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    
    stock_dimension = len(trade.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}, len(INDICATORS): {len(INDICATORS)}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    # please do not change initial_amount
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    check_and_make_directories([TRAINED_MODEL_DIR])

    # Environment
    e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)
    
    # PPO agent
    agent = DRLAgent(env = e_trade_gym)
    model_ppo = agent.get_model("ppo", model_kwargs = PPO_PARAMS)
    trained_ppo = PPO.load(os.path.join(TRAINED_MODEL_DIR, 'trained_ppo'))

    # Backtesting
    df_result_ppo, df_actions_ppo = DRLAgent.DRL_prediction(model=trained_ppo, environment = e_trade_gym)
    df_result_ppo.to_csv(os.path.join(RESULTS_DIR, 'results.csv'), index=False)
    df_actions_ppo.to_csv(os.path.join(RESULTS_DIR, 'actions.csv'), index=False)
    """Plotting"""
    plt.rcParams["figure.figsize"] = (15,5)
    plt.figure()
    
    df_result_ppo.plot()
    plt.savefig(os.path.join(RESULTS_DIR, 'plot.png'))
    # print("==============Get Backtest Results===========")
    # perf_stats_all = backtest_stats(account_value=df_result_ppo)
