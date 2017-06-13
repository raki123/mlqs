import pandas as pd
from PythonCode.util.VisualizeDataset import VisualizeDataset

data_viz = VisualizeDataset()
data = pd.read_csv("./PythonCode/intermediate_datafiles/chapter4_result.csv", index_col = 0)
data.index = data.index.to_datetime()
'''for sensor in ['acc_phone_x_', 'gyr_phone_x_', 'mag_phone_x_']:
	subplots1 = [ sensor + 'freq_0.' + str(i) + "_Hz_ws_40" for i in range(1,10)]
	subplots2 = [ sensor + 'freq_1.' + str(i) + "_Hz_ws_40" for i in range(0,10)]
	subplots2.append(sensor + 'freq_2.0_Hz_ws_40')
	subplots1.append('label')
	subplots2.append('label')
	exact12 = [ 'like' for i in range(12) ]
	line10 = ['line' for i in range(9) ]
	line10.append('points')
	line11 = ['line' for i in range(11) ]
	line11.append('points')
	data_viz.plot_dataset(data, subplots1, exact12, line10)
	data_viz.plot_dataset(data, subplots2, exact12, line11)'''
for sensor in ['acc_phone_x_', 'gyr_phone_x_', 'mag_phone_x_']:
	subplots1 = [ sensor + 'freq_0.9_Hz_ws_40', sensor + 'freq_1.0_Hz_ws_40', sensor + 'freq_1.4_Hz_ws_40', sensor + 'freq_1.5_Hz_ws_40']
	subplots1.append('label')
	exact12 = [ 'like' for i in range(12) ]
	line10 = ['line' for i in range(4) ]
	line10.append('points')
	data_viz.plot_dataset(data, subplots1, exact12, line10)

#data_viz.plot_dataset(data, ['acc_phone_x_freq_0', 'gyr_phone_x_freq_0', 'mag_phone_x_freq_0', 'acc_phone_x_freq_1', 'gyr_phone_x_freq_1', 'mag_phone_x_freq_1', 'acc_phone_x_freq_2', 'gyr_phone_x_freq_2','mag_phone_x_freq_2', 'label'], ['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like','like', 'like'], ['line', 'line','line', 'line', 'line', 'line','line', 'line', 'line', 'points'])
#data_viz.plot_dataset(data, ['acc_phone_x_max_freq', 'acc_phone_x_freq_weighted', 'acc_phone_x_pse', 'label'], ['like', 'like', 'like', 'like'], ['line', 'line', 'line','points'])
#data_viz.plot_dataset(data, ['gyr_phone_x_max_freq', 'gyr_phone_x_freq_weighted', 'gyr_phone_x_pse', 'label'], ['like', 'like', 'like', 'like'], ['line', 'line', 'line','points'])
#data_viz.plot_dataset(data, ['mag_phone_x_max_freq', 'mag_phone_x_freq_weighted', 'mag_phone_x_pse', 'label'], ['like', 'like', 'like', 'like'], ['line', 'line', 'line','points'])
