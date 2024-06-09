import pandas as pd
import matplotlib.pyplot as plt
import argparse

from train import INDICATORS
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import PPO
from finrl.main import check_and_make_directories
from finrl.config import TRAINED_MODEL_DIR
from finrl.plot import backtest_stats
import torch
from kan import KAN
import numpy as np

# Contestants are welcome to split the data in their own way for model tuning
TRADE_START_DATE = '2024-05-01'
TRADE_END_DATE = '2024-05-15'
FILE_PATH = 'datasets/trade_data.csv'
# INDICATORS = [ "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma"]
# "money","stock","close"
HIDDEN_WIDTH=(3,)
# PPO configs
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0003,
    "batch_size": 128,
}

class InterpretablePolicyExtractor:
    lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']

    def __init__(self, obs_dim, act_dim, hidden_widths,device):
        self.device=device
        print([obs_dim, *hidden_widths, act_dim])
        self.policy = KAN(width=[obs_dim, *hidden_widths, act_dim],device=self.device)
        self.loss_fn = torch.nn.MSELoss() 

    def train_from_dataset(self, dataset, steps: int = 20):
        return self.policy.train(dataset, opt="LBFGS", steps=steps, loss_fn=self.loss_fn)

    def forward(self, observation):
        observation = torch.from_numpy(observation).float()
        action = self.policy(observation.unsqueeze(0))
        return action.squeeze(0).detach().numpy()

if __name__ == '__main__':
    # We will use unseen, post-deadline data for testing
    parser = argparse.ArgumentParser(description='Description of program')
    parser.add_argument('-s', '--start_date', default=TRADE_START_DATE, help='Trade start date (default: {})'.format(TRADE_START_DATE))
    parser.add_argument('-e', '--end_date', default=TRADE_END_DATE, help='Trade end date (default: {})'.format(TRADE_END_DATE))
    parser.add_argument('-p', '--file_path', default=FILE_PATH, help='Trade data file')

    # args = parser.parse_known_args()[0]
    args = parser.parse_args()
    TRADE_START_DATE = args.start_date
    TRADE_END_DATE = args.end_date
    device ='cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    processed_full = pd.read_csv(args.file_path)
    trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    
    stock_dimension = len(trade.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

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
    trained_ppo = PPO.load(TRAINED_MODEL_DIR + '/trained_ppo')

    # Backtesting
    e_trade_gym.reset()
    obs, info = e_trade_gym.reset()
    Obs,Act = [], []
    while True:
        act, _ = trained_ppo.predict(obs,deterministic=True)
        obs, reward, done, info, _=e_trade_gym.step(act)
        Obs.append(np.array(obs))
        Act.append(np.array(act))
        if done:
            break
    
    Obs=np.array(Obs)
    Act=np.array(Act)
    stocks=trade.tic.unique()
    stocks_Obs=[np.concatenate((Obs[:,0].reshape(-1,1),Obs[:,i+1:state_space:stock_dimension]),axis=1) for i in range(stock_dimension)]
    stocks_Act=[Act[:,i].reshape(-1,1) for i in range(stock_dimension)]

    input_names=["money","close","holding",*INDICATORS]

    for stock,obs,act in zip(stocks,stocks_Obs,stocks_Act):
        obsTen=torch.tensor(obs).float().to(device)
        actTen=torch.tensor(act).float().to(device)
        dataset={'train_input': obsTen,
                'train_label': actTen,
                'test_input': obsTen,
                'test_label': actTen,
                }
        agent=InterpretablePolicyExtractor(obs_dim=obs.shape[1],act_dim=act.shape[1],hidden_widths=HIDDEN_WIDTH,device=device)
        agent.train_from_dataset(dataset,steps=50)
        agent.policy.prune(threshold=0.0001)
        agent.policy.plot(scale=10, beta=100, title=f'{stock} KAN')#in_vars=input_names,
        print(stock)
        plt.savefig(f"pics/{stock}-kan-policy.png")
        