# Data cleaning and preprocessing class

"""
Involves the following steps:
1. Handling missing values
2. Handling duplicate values
3. Handling non-numerical values
"""

import pandas as pd


class DataCleaning:
    def __init__(self, df):
        self.df = df

    # function to clean data
    def clean(self):
        self.df.iloc[:, 2] = self.df.iloc[:, 2].str.replace('.', '')
        self.df.iloc[:, 3] = self.df.iloc[:, 3].str.replace('.', '')
        self.df.iloc[:, 5] = self.df.iloc[:, 5].str.replace('.', '') 


if __name__ == "__main__":
    df = pd.read_csv("final_dataset.csv")
    data_cleaning = DataCleaning(df)
    data_cleaning.clean()
    df.to_csv("final_dataset_1.csv", index=False)