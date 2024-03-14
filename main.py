from datainsights.Insight import Insights
import pandas as pd
from datacleaner.cleaner import DataCleaning


df = pd.read_csv("final_dataset.csv")
print(df)
data_cleaning = DataCleaning(df)
print("After data cleaning")
data_cleaning.clean()
print("After handling missing values")
df.to_csv("final_dataset_1.csv", index=False)