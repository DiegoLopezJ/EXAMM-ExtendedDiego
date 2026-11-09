import os
from pathlib import Path
import pandas as pd
import optuna

class Profile:
    def __init__(self):
        self.MoneySpend = 0
        self.MoneyEarn = 0
        self.ProfileNames = []
        
    def reset(self):
        self.MoneySpend = 0
        self.MoneyEarn = 0
        self.ProfileNames = []
        
    def add_spend(self, money):
        self.MoneySpend += money
    
    def add_earn(self, money):
        self.MoneyEarn += money
    
    def get_return(self, return_type):
        profile_return = ((self.MoneyEarn - self.MoneySpend) / self.MoneySpend) * 100
        self.reset()
        # print(f"Profile return for {return_type} is {profile_return}")
        return profile_return
    
    def is_in_profile(self, stock_name):
        return stock_name in self.ProfileNames
    
    def add_stock(self, stock_name):
        if not self.is_in_profile(stock_name):
            self.ProfileNames.append(stock_name)
        else:
            print(f"{stock_name} is already in the profile")
    
    def remove_stock(self, stock_name):
        if self.is_in_profile(stock_name):
            self.ProfileNames.remove(stock_name)
        # else:
        #     print(f"{stock_name} is not in the profile")
    
    def simple_return(self, predicted_return, price, stock_name, share=1):
        if(len(predicted_return) + 1 != len(price)):
            print('length of predicted_return is not equal to price')
            exit()
        stock_spend = 0
        stock_earn = 0
        # add it to profile first
        # self.add_stock(stock_name)
        # self.add_spend(price[0] * share)
        # stock_spend += price[0] * share
        # bought_price = price[0]
        for i in range(len(predicted_return)-1):
            if predicted_return[i] > 0:
                if not self.is_in_profile(stock_name):
                    self.add_stock(stock_name)
                    self.add_spend(price[i] * share)
                    stock_spend += price[i] * share
                    bought_price = price[i]
            else:
                if self.is_in_profile(stock_name):
                    # if bought_price < price[i]:
                    if (price[i] - bought_price)/bought_price * 100 > 7:
                        # print(f"Stock {stock_name}, bought price: {bought_price}, sold price: {price[i]}")
                        self.add_earn(price[i] * share)
                        self.remove_stock(stock_name)
                        stock_earn += price[i] * share
        if self.is_in_profile(stock_name):
            self.add_earn(price[-2] * share)
            self.remove_stock(stock_name)
            stock_earn += price[-2] * share
        # stock_return = ((stock_earn - stock_spend) / stock_spend) * 100
        # print(f"Stock return for {stock_name} is {stock_return}")
    
    def long_short_return(self, predicted_return, price, stock_name, share=1):
        if(len(predicted_return) + 1 != len(price)):
            print('length of predicted_return is not equal to price')
            exit()
        # add it to profile first
        self.add_stock(stock_name)
        self.add_spend(price[0])
        for i in range(1, len(predicted_return)-1):
            if predicted_return[i] > 0:
                if not self.is_in_profile(stock_name):
                    print("This should never happen for long short strategy, stock: ", stock_name )
                    self.add_stock(stock_name)
                    self.add_spend(price[i] * share)
            else:
                if self.is_in_profile(stock_name):
                    self.add_earn(price[i] * share)
                    self.remove_stock(stock_name)
                    self.add_spend(price[i+1] * share)
                    self.add_stock(stock_name)
        # self.remove_stock(stock_name)
        if self.is_in_profile(stock_name):
            self.add_earn(price[-2] * share)
            self.remove_stock(stock_name)
    def objective(self,trial):
        share_array = [trial.suggest_int(f'x{i}', 0, 100) for i in range(self.num_stocks)]
        # Suggested value for 'share' to be optimized

        # if self.MoneyEarn != 0 or self.MoneySpend != 0:
        #     print("Error: MoneyEarn and MoneySpend should be 0")
        #     exit()
        for i in range(27):
            # if i != 0:
            #     if self.MoneyEarn == 0 and self.MoneySpend == 0:
            #         print("Error: MoneyEarn and MoneySpend should not be 0")
            #         exit()
            stock_name = stock_names[i]
        # for stock_name in stock_names:ssss
            # print(f"Processing {stock_name}")
            prediction_file = prediction_folder / f"{stock_name}_predictions.csv"
            test_file = test_file_path / f"{stock_name}.csv"
            prediction = pd.read_csv(prediction_file, usecols=['predicted_RET']).to_numpy().flatten()
            stock_price = pd.read_csv(test_file, usecols=['PRC']).to_numpy().flatten()
            
            if self.return_strategy == 'simple_return':
                self.simple_return(prediction, stock_price, stock_name, share_array[i])
            elif self.return_strategy == 'long_short_return':
                self.long_short_return(prediction, stock_price, stock_name, share_array[i])
        return self.get_return('simple_return')


if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parent
    results_root = Path(os.getenv("FIN_RESULTS_DIR", repo_root / "results"))
    datasets_root = Path(os.getenv("FIN_DATASETS_DIR", repo_root / "datasets"))

    prediction_folder = Path(os.getenv("FIN_PREDICTION_DIR", results_root / "0216" / "predictions"))
    test_file_path = Path(os.getenv("FIN_TEST_DIR", datasets_root / "CRSP" / "DJI_history_new" / "single_stocks" / "test"))
    stock_names = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DOW', 'DIS', 'WBA', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',  'PG', 'TRV', 'UNH',  'VZ', 'V', 'WMT']
    # stock_names = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DOW', 'DIS', 'WBA', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',  'PG', 'TRV', 'UNH',  'VZ', 'V', 'WMT', 'HON', 'AMZN', 'CRM']

    profile = Profile()
    profile.num_stocks = len(stock_names)
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    # return_strategy = 'long_short_return'
    for return_strategy in ["simple_return"]:
        print(f"Optimizing {return_strategy} strategy")
        profile.reset()
        profile.return_strategy = return_strategy
        study = optuna.create_study(direction='maximize')
        study.optimize(profile.objective, n_trials=1000)  # You can adjust the number of trials

        print(f"Best trial: {study.best_trial.value}")
        # print(f"Optimal share: {study.best_trial.params['share']}")
