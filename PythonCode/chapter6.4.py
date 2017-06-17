##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.LearningAlgorithms import RegressionAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.Evaluation import RegressionEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from Chapter7.FeatureSelection import FeatureSelectionRegression
import copy
import pandas as pd
from util import util
import matplotlib.pyplot as plot
import numpy as np
from sklearn.model_selection import train_test_split
import os


# Of course we repeat some stuff from Chapter 3, namely to load the dataset

DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sure the index is of the type datetime.

dataset_path = './intermediate_datafiles/'
export_tree_path = 'Example_graphs/Chapter7/'

try:
    dataset = pd.read_csv(dataset_path + 'chapter5_result.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

if not os.path.exists(export_tree_path):
    os.makedirs(export_tree_path)

dataset.index = dataset.index.to_datetime()

dataset1 = pd.read_csv('./intermediate_datafiles/smallncomparedtop.csv', index_col = 0)
dataset1.index = dataset1.index.to_datetime()

for dataset in [dataset, dataset1]:

	# Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.

	# We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
	# for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
	# cases where we do not know the label.

	prepare = PrepareDatasetForLearning()

	train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.7, filter=True, temporal=False)

	print 'Training set length is: ', len(train_X.index)
	print 'Test set length is: ', len(test_X.index)

	# Select subsets of the features that we will consider:

	basic_features = ['acc_phone_x','acc_phone_y','acc_phone_z','acc_watch_x','acc_watch_y','acc_watch_z','gyr_phone_x','gyr_phone_y','gyr_phone_z','gyr_watch_x','gyr_watch_y','gyr_watch_z',
			  'hr_watch_rate', 'light_phone_lux','mag_phone_x','mag_phone_y','mag_phone_z','mag_watch_x','mag_watch_y','mag_watch_z','press_phone_pressure']
	pca_features = ['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7']
	time_features = [name for name in dataset.columns if '_temp_' in name]
	freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
	print '#basic features: ', len(basic_features)
	print '#PCA features: ', len(pca_features)
	print '#time features: ', len(time_features)
	print '#frequency features: ', len(freq_features)
	cluster_features = ['cluster']
	print '#cluster features: ', len(cluster_features)
	features_after_chapter_3 = list(set().union(basic_features, pca_features))
	features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
	features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features, cluster_features))


	# First, let us consider the performance over a selection of features:


	learner = ClassificationAlgorithms()
	eval = ClassificationEvaluation()


	# And we study two promising ones in more detail. First let us consider the decision tree which works best with the selected
	# features.
	#
	class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(train_X[features_after_chapter_5], train_y, test_X[features_after_chapter_5],
												   gridsearch=True,
												   print_model_details=True, export_tree_path=export_tree_path)

	#class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(train_X[features_after_chapter_5], train_y, test_X[features_after_chapter_5],
	#                                                                                           gridsearch=True, print_model_details=True)

	test_cm = eval.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)

	DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False)

	reg_parameters = [0.0001, 0.001, 0.01, 0.1, 1, 10]
	performance_training = []
	performance_test = []

	# We repeat the experiment a number of times to get a bit more robust data as the initialization of the NN is random.

	repeats = 20
	for reg_param in reg_parameters:
		performance_tr = 0
		performance_te = 0
    		for i in range(0, repeats):
        		class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(train_X, train_y, test_X, hidden_layer_sizes=(250, ), alpha=reg_param, max_iter=500,gridsearch=False)
			performance_tr += eval.accuracy(train_y, class_train_y)
			performance_te += eval.accuracy(test_y, class_test_y)
		performance_training.append(performance_tr/repeats)
		performance_test.append(performance_te/repeats)
		
	plot.hold(True)
	plot.semilogx(reg_parameters, performance_training, 'r-')
	plot.semilogx(reg_parameters, performance_test, 'b:')
	print performance_training
	print performance_test
	plot.xlabel('regularization parameter value')
	plot.ylabel('accuracy')
	plot.ylim([0.95, 1.01])
	plot.legend(['training', 'test'], loc=4)
	plot.hold(False)
	plot.show()
