import pandas as pd
import os

# Quick test read
df_demographics = pd.read_parquet("~/Documents/Internship_INSPECT/DATA_RAW/EHR/person/person.parquet")
print(df_demographics.head())