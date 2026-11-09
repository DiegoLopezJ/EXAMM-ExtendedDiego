import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error 

# data_path = ''
# data = pd.read_csv(data_path)

# expected = data.iloc[:, 1]
# predicted = data.iloc[:, 2]

# mse = mean_squared_error(expected, predicted)
# print("Mean Squared Error of prediction: {:.6f}".format(mse))

# mse_naive = mean_squared_error(expected[1:], expected[:-1])
# print("Mean Squared Error of naive prediction: {:.6f}".format(mse_naive))

# files = glob(data_dir + "/*.csv")
# overall_mse = []
# for file in files:
#     data = pd.read_csv(file)
#     expected = data["expected_RET"]
#     predicted = data["predicted_RET"]

#     mse = mean_squared_error(expected, predicted)
#     print("Mean Squared Error of prediction: {:.6f}".format(mse))
#     overall_mse.append(mse)
# print("Overall Mean Squared Error: {:.6f}".format(sum(overall_mse) / len(overall_mse)))
repo_root = Path(__file__).resolve().parent
datasets_root = Path(os.getenv("FIN_DATASETS_DIR", repo_root / "datasets"))

stock_names = ['AAPL', 'AMZN', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DOW', 'DIS', 'WBA', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',  'PG', 'TRV', 'UNH',  'VZ', 'V', 'WMT',  'HON', 'CRM']
prediction_dir = Path(os.getenv("FIN_PREDICTION_DIR", repo_root / "predictions"))
expected_dir = Path(os.getenv("FIN_EXPECTED_DIR", datasets_root / "CRSP" / "DJI_company_2023" / "test"))
overall_mse = []
for stock in stock_names:
    prediction_file = prediction_dir / f"{stock}.csv"
    expected_file = expected_dir / f"{stock}.csv"
    # files = glob(data_dir + "/*.csv")
    expected_data = pd.read_csv(expected_file)
    prediction_data = pd.read_csv(prediction_file)

    expected = expected_data["RET"]
    predicted = prediction_data["Predicted_RET"]

    mse = mean_squared_error(expected, predicted)
    print("Mean Squared Error of prediction: {:.6f}".format(mse))
    overall_mse.append(mse)
print("Overall Mean Squared Error: {:.6f}".format(sum(overall_mse) / len(overall_mse)))
