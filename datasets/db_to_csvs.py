import sys
import pandas as pd

sensor = sys.argv[1]

new_data = pd.read_csv(sensor + ".csv")
print new_data
new_data['evald'] = map(eval, new_data[2])
if type(new_data['evald'][0]) is list:	
	vals = []
	for i in range(len(new_data['evald'][0])):
		vals.append(map(lambda x: x[0], new_data['evald']))
	columns = ["timestamp"]
	columns.extend([sensor + "_" + str(i) for i in range(len(new_data['evald'][0]))])
	print_data = pd.DataFrame(columns = columns)
	print_data['timestamp'] = new_data['timestamp']*1000000
	for i in range(len(new_data['evald'][0])):
		print_data[sensor + "_" + str(i)] = vals[i]
	print_data.to_csv(sensor + "_out.csv")
else:
	print_data = pd.DataFrame(columns = [sensor, "timestamp"])
	print_data['timestamp'] = new_data['timestamp']*1000000
	print_data[sensor] = new_data['evald']
	print_data.to_csv(sensor + "_out.csv")
	
