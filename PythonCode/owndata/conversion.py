import sys
import pandas as pd

sensor = sys.argv[1]

new_data = pd.read_csv(sensor)
print new_data
new_data['evald'] = map(eval, new_data["value"])
if type(new_data['evald'][0]) is list:
        vals = []
        for i in range(len(new_data['evald'][0])):
                vals.append(map(lambda x: x[i], new_data['evald']))
        columns = ["timestamp"]
        columns.extend([str(chr(i+ord('x'))) for i in range(len(new_data['evald'][0]))])
        print_data = pd.DataFrame(columns = columns)
        tsmps = []
	for i in new_data['timestamp']:
		if (i > 1497399630783):
			tsmps.append((i - (1497487993722-1497399630783))*1000000)
		else:
			tsmps.append(i*1000000)
	print_data['timestamp'] = tsmps
	for i in range(len(new_data['evald'][0])):
                print_data[str(chr(i+ord('x')))] = vals[i]
        print_data.to_csv(sensor + "_out.csv")
else:
        print_data = pd.DataFrame(columns = [sensor, "timestamp"])
        print_data['timestamp'] = new_data['timestamp']*1000000
        print_data[sensor] = new_data['evald']
        print_data.to_csv(sensor + "_out.csv")
