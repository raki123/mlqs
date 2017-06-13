import pandas as pd
from PythonCode.Chapter3.ImputationMissingValues import ImputationMissingValues
from PythonCode.Chapter3.KalmanFilters import KalmanFilters
from PythonCode.util.VisualizeDataset import VisualizeDataset
import copy

data_viz = VisualizeDataset()
imputer = ImputationMissingValues()
data = pd.read_csv("./PythonCode/intermediate_datafiles/chapter2_result.csv", index_col=0)
data.index = data.index.to_datetime()

data_imputed_linear = imputer.impute_interpolate(copy.deepcopy(data), 'hr_watch_rate')
KalFilter = KalmanFilters()
kalman_dataset = KalFilter.apply_kalman_filter(data, 'hr_watch_rate')
data_viz.plot_imputed_values(data, ['original','interpolation', 'kalman'], 'hr_watch_rate', data_imputed_linear['hr_watch_rate'], kalman_dataset['hr_watch_rate_kalman'])
