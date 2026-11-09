import csv
import math
import os
from pathlib import Path

# Path to your CSV file
repo_root = Path(__file__).resolve().parent
datasets_root = Path(os.getenv("FIN_DATASETS_DIR", repo_root / "datasets"))
csv_file_path = datasets_root / "CRSP" / "DJI_company_2023" / "validation" / "TRV.csv"

with open(csv_file_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    
    # Collect rows with inf values
    rows_with_inf = []
    for row_number, row in enumerate(csv_reader):
        # Check if any cell in the row contains 'inf'
        if any(math.isinf(float(cell)) if cell.replace('.', '', 1).lstrip('-+').isdigit() else False for cell in row):
            rows_with_inf.append((row_number, row))

for row_number, row in rows_with_inf:
    print(f"Row {row_number} contains inf: {row}")
