##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################


dataset_path = '/Users/mcjvasseur/surfdrive/Studie/Master/Semester2/ML4QS/owndata/'
result_dataset_path = './intermediate_datafiles-own/'

# Import the relevant classes.

from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
import copy
import os


if not os.path.exists(result_dataset_path):
    print('Creating result directory: ' + result_dataset_path)
    os.makedirs(result_dataset_path)

# Chapter 2: Initial exploration of the dataset.

# Set a granularity (i.e. how big are our discrete time steps). We start very
# coarse grained, namely one measurement per minute, and secondly use four measurements
# per second

granularities = [250]
datasets = []

for milliseconds_per_instance in granularities:

    # Create an initial dataset object with the base directory for our data and a granularity
    DataSet = CreateDataset(dataset_path, milliseconds_per_instance)

    # Add the selected measurements to it.

    # We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values/
    DataSet.add_numerical_dataset('accelerometer-kx023.csv_out.csv', 'timestamp', ['x','y','z'], 'avg', 'acc_phone_')
    print("first set")
    # We add the gyroscope data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values/
    DataSet.add_numerical_dataset('orientation.csv_out.csv', 'timestamp', ['x','y','z'], 'avg', 'gyr_phone_')
    print("second set")
    # We add the labels provided by the users. These are categorical events that might overlap. We add them
    # as binary attributes (i.e. add a one to the attribute representing the specific value for the label if it
    # occurs within an interval).
    DataSet.add_event_dataset('status.csv', 'timestampBeg', 'timestampEnd', 'label', 'binary')

    # We add the amount of light sensed by the phone (continuous numerical measurements) and aggregate by averaging again
    DataSet.add_numerical_dataset('light-bh1745.csv_out.csv', 'timestamp', ['lux'], 'avg', 'light_phone_')
    print("third set")
    # We add the magnetometer data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values
    DataSet.add_numerical_dataset('mag-akm09911.csv_out.csv', 'timestamp', ['x','y','z'], 'avg', 'mag_phone_')
    print("fourth set")
    # We add the pressure sensed by the phone (continuous numerical measurements) and aggregate by averaging again

    # Get the resulting pandas data table

    dataset = DataSet.data_table
    print(dataset)
    # Plot the data

    DataViz = VisualizeDataset()

    # Boxplot
    #DataViz.plot_dataset_boxplot(dataset, ['acc_phone_accelerometer-kx023.csv_0', 'acc_phone_accelerometer-kx023.csv_1','acc_phone_accelerometer-kx023.csv_2'])

    # Plot all data
    #DataViz.plot_dataset(dataset, ['acc_phone_','name', 'mag_phone_','light_phone_','gyr_phone_'], ['like','like','like','like','like'], ['line','points','line','line','line'])

    # And print a summary of the dataset

    util.print_statistics(dataset)
    datasets.append(copy.deepcopy(dataset))

# And print the table that has been included in the book

#util.print_latex_table_statistics_two_datasets(datasets[0],datasets[1])

# Finally, store the last dataset we have generated (500 ms).
dataset.to_csv(result_dataset_path + 'chapter2_result-own.csv')

their = dataset.from_csv('./intermediate_datafiles/' + 'chapter2_result.csv')
util.print_latex_table_statistics_two_datasets(dataset,their)
