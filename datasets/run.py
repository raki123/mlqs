from PythonCode.Chapter2 import CreateDataset
import pandas as pd
import datetime

ds = CreateDataset.CreateDataset("", 10000)
ds.create_dataset(pd.to_datetime(1497392462424, unit = 'ms'),pd.to_datetime(1497399630773, unit = 'ms'),[], '')
ds.add_numerical_dataset("../out_out.csv","timestamp", ["out_0","out_1","out_2"])
ds.add_numerical_dataset("../out_out.csv","timestamp", ["out_0","out_1","out_2"], prefix = "sec")
print ds.data_table

