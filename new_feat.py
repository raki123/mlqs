import pandas as pd
from PythonCode.util.VisualizeDataset import VisualizeDataset
import numpy as np
from PythonCode.Chapter4.TemporalAbstraction import NumericalAbstraction

data_viz = VisualizeDataset()
data = pd.read_csv("./PythonCode/intermediate_datafiles/chapter4_result.csv", index_col = 0)
data.index = data.index.to_datetime()
#time
milliseconds_per_instance = (data.index[1] - data.index[0]).microseconds/1000
window_sizes = [int(float(5000)/milliseconds_per_instance), int(float(0.5*60000)/milliseconds_per_instance), int(float(5*60000)/milliseconds_per_instance)]
NumAbs = NumericalAbstraction()
for ws in window_sizes:
	data = NumAbs.abstract_numerical(data, ['acc_phone_x'], ws, 'max')
	data = NumAbs.abstract_numerical(data, ['acc_phone_x'], ws, 'slope')
subplots1 = ['acc_phone_x', 'acc_phone_x_temp_max', 'acc_phone_x_temp_slope', 'label']
exact12 = [ 'exact', 'like', 'like', 'like' ]
line10 = ['line' for i in range(3) ]
line10.append('points')
data_viz.plot_dataset(data, subplots1, exact12, line10)

#frequency
#energy
squares = [ map(lambda x: x*x,data['acc_phone_x_freq_0.' + str(i) + '_Hz_ws_40']) for i in range(1,10) ]
squares.extend([ map(lambda x: x*x,data['acc_phone_x_freq_1.' + str(i) + '_Hz_ws_40']) for i in range(1,10) ])
energy = map(lambda x: x*x, data['acc_phone_x_freq_2.0_Hz_ws_40'])
for arr in squares:
	for i in range(len(arr)):
		energy[i] += arr[i]
energy = np.dot(energy,0.05)
data['energy'] = energy
#energyband
squares = [ map(lambda x: x*x,data['acc_phone_x_freq_0.' + str(9) + '_Hz_ws_40']),map(lambda x: x*x,data['acc_phone_x_freq_1.0_Hz_ws_40'])]
energy = np.zeros([len(energy)])
for arr in squares:
	for i in range(len(arr)):
		energy[i] += arr[i]
energy = np.dot(energy,0.5)
data['energy_walking'] = energy

squares = [ map(lambda x: x*x,data['acc_phone_x_freq_1.4_Hz_ws_40']),map(lambda x: x*x,data['acc_phone_x_freq_1.5_Hz_ws_40'])]
energy = np.zeros([len(energy)])
for arr in squares:
	for i in range(len(arr)):
		energy[i] += arr[i]
energy = np.dot(energy,0.5)
data['energy_running'] = energy
#max
squares = [ data['acc_phone_x_freq_0.' + str(i) + '_Hz_ws_40'] for i in range(1,10) ]
squares.extend([data['acc_phone_x_freq_1.' + str(i) + '_Hz_ws_40'] for i in range(1,10) ])
squares.append(data['acc_phone_x_freq_2.0_Hz_ws_40'])
max_amp = np.zeros([len(energy)])-1000000000
min_amp = np.zeros([len(energy)])+1000000000
for arr in squares:
	for i in range(len(arr)):
		max_amp[i] = max([arr[i], max_amp[i]])
		min_amp[i] = min([arr[i], min_amp[i]])
data['max_amplitude'] = max_amp
data['min_amplitude'] = min_amp
subplots1 = ['energy', 'energy_walking', 'energy_running', 'max_amplitude', 'min_amplitude']
subplots1.append('label')
exact12 = [ 'exact', 'exact', 'exact', 'exact', 'exact', 'like' ]
line10 = ['line' for i in range(5) ]
line10.append('points')
data_viz.plot_dataset(data, subplots1, exact12, line10)
