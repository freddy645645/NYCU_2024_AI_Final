# NYCU_2024_AI_Final
![Python](https://img.shields.io/badge/python-3670A0?&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-blue)
![FinRL](https://img.shields.io/badge/FinRL-purple)
![yfinance](https://img.shields.io/badge/yfinance-blueviolet)
![PPO](https://img.shields.io/badge/PPO-blue)
![KAN](https://img.shields.io/badge/KAN-blue)

Our topic is utilizing Kolmogorov-Arnold Networks (KAN) and feature engineering to enhance the performance of reinforcement learning (RL) in the FinRL environment. FinRL is an RL environment from NIPS Datasets and Benchmarks 2022, providing benchmarks for researching RL algorithms in financial markets.

## Getting Started

#### Setup Environment

```bash
git clone https://github.com/freddy645645/NYCU_2024_AI_Final.git
cd NYCU_2024_AI_Final
pip install -r requirements.txt
```

#### Fetch Data

Use the following command to fetch data from Yahoo Finance.

You can use `--help` to see the help message of the script.

```bash
python get_data.py --train -s 2024-01-01 -e 2024-04-30
python get_data.py -s 2024-05-01 -e 2024-05-15
```

#### Train the model

We have provided a trained model in the `trained_models` folder. If you want to train the model with your data, you can modify the constants in the `train.py` file. Then you can train the model by running the following command.
```bash
python train.py
```

#### Visualize the trained model

You can visualize the trained model by running the following command. The output will be saved in the `results/pic` folder. You can use `--help` to see the help message of the script.

```bash
python interpretable.py
```

#### Backtest the model

You can backtest the model of your testing data by running the following command. The output will be saved in the `results` folder. You can use `--help` to see the help message of the script.
```bash
python test.py
```

## Comparison to baseline
We use the original model of FinRL as the baseline

<img src = "https://github.com/freddy645645/NYCU_2024_AI_Final/assets/118954765/c073cd26-7792-4b2b-8de5-775f4dbbb005"/>
