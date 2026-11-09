import os
from pathlib import Path
import pandas as pd
import shutil

rnn_type = 'gru'
year = 2022
repo_root = Path(__file__).resolve().parent
results_root = Path(os.getenv("FIN_RESULTS_DIR", repo_root / "results"))
exact_root = Path(os.getenv("EXACT_RESULTS_DIR", repo_root / "exact"))

prediction_path = results_root / f"{rnn_type}_prediction_{year}"

best_mse = 100
overall_mse = []
if not prediction_path.exists():
    prediction_path.mkdir(parents=True, exist_ok=True)
    print("Created directory: {}".format(prediction_path))

for stock in ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DOW', 'DIS', 'WBA', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',  'PG', 'TRV', 'UNH',  'VZ', 'V', 'WMT', 'HON', 'AMZN', 'CRM']:
    print("Processing stock: {}".format(stock))
    best_mse = 100
    for i in range(10):
        fitness_filepath = exact_root / "DJI_Company" / f"{year}_{rnn_type}" / stock / str(i)
        if fitness_filepath.exists() and fitness_filepath.is_dir():

            df = pd.read_csv(fitness_filepath / "fitness.csv")
            mse = df[' BEST Val. MSE'].iloc[-1]
            if mse < best_mse:
                best_mse = mse
                prediction_file_path = fitness_filepath / f"{stock}_test_predictions.csv"
    overall_mse.append(best_mse)
    prediction_file = prediction_path / f"{stock}_predictions.csv"
    shutil.copy(prediction_file_path, prediction_file)
    print("copied {} to {}".format(prediction_file_path, prediction_file))

print("Overall best MSE: {}".format(sum(overall_mse) / len(overall_mse)))
    
