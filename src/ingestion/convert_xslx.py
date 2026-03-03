import pandas as pd
import os

def convert_xlsx_to_csv(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")

    df = pd.read_excel(input_path)
    df.to_csv(output_path, index=False)
    print(f"Saved CSV to {output_path}")

if __name__ == "__main__":
    convert_xlsx_to_csv(
        input_path="data/raw/Online Retail.xlsx",
        output_path="data/raw/online_retail.csv"
    )