import os
from pathlib import Path
import pandas as pd
from portfolio import Portfolio
from stock import Stock
from Logger import Logger
import csv

if __name__ == '__main__':
    year = 2022
    rnn_type = 'rf'
    print("Testing on year: ", year)

    repo_root = Path(__file__).resolve().parent
    results_root = Path(os.getenv("FIN_RESULTS_DIR", repo_root / "results"))
    crsp_processor_root = Path(os.getenv("CRSP_PROCESSOR_DIR", repo_root / "CRSP_Processor"))

    PREDICTION_DIR = results_root / f"2022_sp_500_{year}_all_prediction"
    TEST_DIR = crsp_processor_root / f"{year}_sp_500" / "test"

    # stock_names = ['AAPL', 'AMZN', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DOW', 'DIS', 'WBA', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',  'PG', 'TRV', 'UNH',  'VZ', 'V', 'WMT',  'HON', 'CRM']
    # stock_names = ['NDSN', 'HOLX', 'ATO', 'KIM', 'NVR', 'DPZ', 'JBHT', 'DECK', 'FRT', 'KMX', 'EXR', 'LNT', 'CINF', 'ED', 'REG', 'CPB', 'LH', 'TRMB', 'DVA', 'PKG', 'CAG', 'NTRS', 'KMB', 'TRV', 'RHI', 'UHS', 'EMN', 'PODD', 'TECH', 'EXPD', 'WRB', 'EIX', 'STLD', 'BXP', 'CHRW', 'IVZ', 'HSIC', 'TFX', 'AKAM', 'JKHY', 'HBAN', 'ESS', 'ETR', 'FFIV', 'CPT', 'IEX', 'IRM', 'COO', 'MHK', 'FDS']
    stock_names = [
        'A', 'AAL', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP',
        'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL',
        'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS',
        'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK',
        'AXON', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF',
        'BG', 'BIO', 'BK', 'BKNG', 'BKR', 'BLDR', 'BLK', 'BMY', 'BR', 'BRK', 'BRO', 'BSX',
        'BWA', 'BX', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI',
        'CCL', 'CDAY', 'CDNS', 'CDW', 'CE', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF',
        'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO',
        'COP', 'COR', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSGP', 'CSX',
        'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DD', 'DE',
        'DECK', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DOW',
        'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX',
        'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT',
        'ES', 'ESS', 'ETN', 'ETR', 'ETSY', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F',
        'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FLT',
        'FMC', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GE', 'GEN', 'GILD', 'GIS',
        'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL',
        'HAS', 'HBAN', 'HCA', 'HCN', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE',
        'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX',
        'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG',
        'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KDP',
        'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'L', 'LDOS',
        'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV',
        'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO',
        'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST',
        'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI',
        'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE',
        'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI',
        'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PARA', 'PAYC',
        'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD',
        'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC',
        'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL',
        'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW',
        'SJM', 'SLB', 'SMCI', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT',
        'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TDG', 'TDY', 'TECH', 'TEL', 'TER',
        'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO',
        'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS', 'ULTA',
        'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VICI', 'VLO', 'VMC', 'VRSK', 'VRSN', 'VRTX',
        'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WDC', 'WEC', 'WFC', 'WM', 'WMB', 'WMT', 'WRB',
        'WRK', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZTS',
    ]

    logger = Logger()
    logger.set_level('DEBUG')

    spend_per_stock = 100
    initial_capital = 1000
    portfolio = Portfolio(stock_names)
    portfolio.set_initial_spend_per_stock(spend_per_stock)
    portfolio.set_initial_capital(initial_capital)
    portfolio.set_logger(logger)

    # return_strategy = 'simple_return'
    # return_strategy = 'portfolio_simple_return'
    # return_strategy = "portfolio_shorting_return"
    return_strategy = 'daily_long_short_return'
    # return_strategy = 'long_short_return'
    # return_strategy = 'daily_long_return'



    for stock_name in stock_names:
        company = Stock(stock_name)
        # print("loaded stock: ", stock_name)
        prediction_file = PREDICTION_DIR / f"{stock_name}_predictions.csv"
        # prediction_file = os.path.join(PREDICTION_DIR, stock_name + ".csv")
        test_file = TEST_DIR / f"{stock_name}.csv"
        company.read_return_prediction(str(prediction_file))
        company.read_stock_price(str(test_file))
        # print("stock {}  got data".format(stock_name))
        portfolio.add_company_to_protfolio(company)
        company.set_logger(logger)

    
    long_company_number=10
    short_company_number=10

    use_TC = False
    trade_with_bid_ask = False
    logger.log(f"Trading with TC: {use_TC}", 'INFO')
    logger.log(f"Trading with bid ask: {trade_with_bid_ask}", 'INFO')
    portfolio.set_use_TC(use_TC)
    portfolio.set_trade_with_bid_ask(trade_with_bid_ask)
    # portfolio.set_waiting_period(5)
    for return_strategy in ['daily_long_short_return' ]:
        portfolio.trade(return_strategy, long_company_number, short_company_number)

    # use_TC = True
    # trade_with_bid_ask = False
    # logger.log(f"Trading with TC: {use_TC}", 'INFO')
    # logger.log(f"Trading with bid ask: {trade_with_bid_ask}", 'INFO')
    # portfolio.set_use_TC(use_TC)
    # portfolio.set_trade_with_bid_ask(trade_with_bid_ask)
    # portfolio.set_waiting_period(5)
    # portfolio.trade(return_strategy, long_company_number, short_company_number)


    # use_TC = False
    # trade_with_bid_ask = True
    # logger.log(f"Trading with TC: {use_TC}", 'INFO')
    # logger.log(f"Trading with bid ask: {trade_with_bid_ask}", 'INFO')
    # portfolio.set_use_TC(use_TC)
    # portfolio.set_trade_with_bid_ask(trade_with_bid_ask)
    # portfolio.set_waiting_period(5)
    # portfolio.trade(return_strategy, long_company_number, short_company_number)

    # return_strategy = 'daily_long_short_return'
    # with open(f'{year}_sp50_hybrid_daily_long_short_500.csv', "wt", encoding="utf-8") as csv_file:
    #     # writer = csv.writer(csv_file)
    #     header = ''
    #     for i in range(1,100):
    #         header += f',short_{i}'
    #     csv_file.write(header + '\n')

    #     # Iterate over all combinations of param1 and param2
    #     for long_company_number in range(1, 100):
    #         row = f'long_{long_company_number},'
    #         for short_company_number in range(1, 100):
    #             portfolio_return = portfolio.trade(return_strategy, long_company_number, short_company_number)
    #             print(f"Portfolio return: {portfolio_return}")
    #             row += f"{portfolio_return:.2f},"
    #         csv_file.write(row + '\n')
    #             # Log the results in the CSV



