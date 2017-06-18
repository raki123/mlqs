##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 8                                               #
#                                                            #
##############################################################

from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.Evaluation import RegressionEvaluation
from Chapter8.LearningAlgorithmsTemporal import TemporalClassificationAlgorithms
from Chapter8.LearningAlgorithmsTemporal import TemporalRegressionAlgorithms
from Chapter7.FeatureSelection import FeatureSelectionRegression
from statsmodels.tsa.stattools import adfuller
from pandas.tools.plotting import autocorrelation_plot

import copy
import sys
import pandas as pd
from util import util
import matplotlib.pyplot as plot
import numpy as np
from sklearn.model_selection import train_test_split

def forward_selection(max_features, X_train, y_train):
    ordered_features = []
    ordered_scores = []
    # Start with no features.
    selected_features = []
    learner = TemporalRegressionAlgorithms()
    eval = RegressionEvaluation()
    prev_best_perf = sys.float_info.max

    # Select the appropriate number of features.
    for i in range(0, max_features):
	print i
        #Determine the features left to select.
        features_left = list(set(X_train.columns) - set(selected_features))
        best_perf = sys.float_info.max
        best_feature = ''

        # For all features we can still select...
        for f in features_left:
            temp_selected_features = copy.deepcopy(selected_features)
            temp_selected_features.append(f)
            # Determine the mse of a decision tree learner if we were to add
            # the feature.
            regr_train_y, regr_test_y = learner.recurrent_neural_network(X_train, y_train, X_train, y_train, gridsearch=False)

            perf, std_tr = eval.mean_squared_error_with_std(train_y.ix[10:,], regr_train_y.ix[10:,])

            # If the performance is better than what we have seen so far (we aim for low mse)
            # we set the current feature to the best feature and the same for the best performance.
            if perf < best_perf:
                best_perf = perf
                best_feature = f
        # We select the feature with the best performance.
        selected_features.append(best_feature)
        prev_best_perf = best_perf
        ordered_features.append(best_feature)
        ordered_scores.append(best_perf)
    return selected_features, ordered_features, ordered_scores

# Of course we repeat some stuff from Chapter 3, namely to load the dataset

DataViz = VisualizeDataset()

dataset_path = './intermediate_datafiles-own/'

try:
    dataset = pd.read_csv(dataset_path + 'chapter5_result-own.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e


dataset.index = dataset.index.to_datetime()
dataset = dataset.dropna()
del dataset['silhouette']

# Let us consider our first task, namely the prediction of the label. We consider this as a non-temporal task.

# We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
# for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
# cases where we do not know the label.

prepare = PrepareDatasetForLearning()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression_by_time(dataset, 'acc_phone_x', '2017-06-13 22:21:02',
#                                                                                   '2016-02-08 18:29:58','2016-02-08 18:29:59')
                                                                                   '2017-06-13 23:40:47', '2017-06-14 00:22:24')

print 'Training set length is: ', len(train_X.index)
print 'Test set length is: ', len(test_X.index)

# Select subsets of the features that we will consider:

print 'Training set length is: ', len(train_X.index)
print 'Test set length is: ', len(test_X.index)

# Select subsets of the features that we will consider:

basic_features = ['acc_phone_y','acc_phone_z','gyr_phone_x','gyr_phone_y','gyr_phone_z','light_phone_lux','mag_phone_x','mag_phone_y','mag_phone_z']
pca_features = ['pca_1','pca_2','pca_3','pca_4']
time_features = [name for name in dataset.columns if ('temp_' in name and not 'acc_phone_x' in name)]
freq_features = [name for name in dataset.columns if ((('_freq' in name) or ('_pse' in name)) and not 'acc_phone_x' in name)]
print '#basic features: ', len(basic_features)
print '#PCA features: ', len(pca_features)
print '#time features: ', len(time_features)
print '#frequency features: ', len(freq_features)
cluster_features = ['cluster']
print '#cluster features: ', len(cluster_features)
features_after_chapter_3 = list(set().union(basic_features, pca_features))
features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features, cluster_features))

#fs = FeatureSelectionRegression()
#selected_features, ordered_features, ordered_scores = fs.forward_selection(10, train_X[features_after_chapter_5], train_y)

#selected_features, ordered_features, ordered_scores = forward_selection(10, train_X, train_y)
#print selected_features
selected_features = ['gyr_phone_z', 'mag_phone_x', 'pca_1', 'acc_phone_y_freq_1.4_Hz_ws_40', 'gyr_phone_y_freq_1.3_Hz_ws_40', 'acc_phone_z_freq_0.9_Hz_ws_40', 'mag_phone_y_freq_0.0_Hz_ws_40', 'mag_phone_y_freq_1.3_Hz_ws_40', 'gyr_phone_z_temp_std_ws_120', 'gyr_phone_y_pse']

possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']

# Let us first study whether the time series is stationary and what the autocorrelations are.

dftest = adfuller(dataset['acc_phone_x'], autolag='AIC')
print dftest

autocorrelation_plot(dataset['acc_phone_x'])
plot.show()

# Now let us focus on the learning part.

learner = TemporalRegressionAlgorithms()
eval = RegressionEvaluation()

# We repeat the experiment a number of times to get a bit more robust data as the initialization of the NN is random.

repeats = 5

# we set a washout time to give the NN's the time to stabilize. We do not compute the error during the washout time.

washout_time = 10

scores_over_all_algs = []
'''
for i in range(len(feature_names)):

    print feature_names[i]
    selected_train_X = train_X[possible_feature_sets[i]]
    selected_test_X = test_X[possible_feature_sets[i]]

    # First we run our non deterministic classifiers a number of times to average their score.

    performance_tr_res = 0
    performance_tr_res_std = 0
    performance_te_res = 0
    performance_te_res_std = 0
    performance_tr_rnn = 0
    performance_tr_rnn_std = 0
    performance_te_rnn = 0
    performance_te_rnn_std = 0

    for repeat in range(0, repeats):
        print '----', repeat
        regr_train_y, regr_test_y = learner.reservoir_computing(selected_train_X, train_y, selected_test_X, test_y, gridsearch=True, per_time_step=False)

        mean_tr, std_tr = eval.mean_squared_error_with_std(train_y.ix[washout_time:,], regr_train_y.ix[washout_time:,])
        mean_te, std_te = eval.mean_squared_error_with_std(test_y.ix[washout_time:,], regr_test_y.ix[washout_time:,])

        performance_tr_res += mean_tr
        performance_tr_res_std += std_tr
        performance_te_res += mean_te
        performance_te_res_std += std_te

        regr_train_y, regr_test_y = learner.recurrent_neural_network(selected_train_X, train_y, selected_test_X, test_y, gridsearch=True)

        mean_tr, std_tr = eval.mean_squared_error_with_std(train_y.ix[washout_time:,], regr_train_y.ix[washout_time:,])
        mean_te, std_te = eval.mean_squared_error_with_std(test_y.ix[washout_time:,], regr_test_y.ix[washout_time:,])

        performance_tr_rnn += mean_tr
        performance_tr_rnn_std += std_tr
        performance_te_rnn += mean_te
        performance_te_rnn_std += std_te


    # We only apply the time series in case of the basis features.
    if (feature_names[i] == 'initial set'):
        regr_train_y, regr_test_y = learner.time_series(selected_train_X, train_y, selected_test_X, test_y, gridsearch=True)

        mean_tr, std_tr = eval.mean_squared_error_with_std(train_y.ix[washout_time:,], regr_train_y.ix[washout_time:,])
        mean_te, std_te = eval.mean_squared_error_with_std(test_y.ix[washout_time:,], regr_test_y.ix[washout_time:,])

        overall_performance_tr_ts = mean_tr
        overall_performance_tr_ts_std = std_tr
        overall_performance_te_ts = mean_te
        overall_performance_te_ts_std = std_te
    else:
        overall_performance_tr_ts = 0
        overall_performance_tr_ts_std = 0
        overall_performance_te_ts = 0
        overall_performance_te_ts_std = 0

    overall_performance_tr_res = performance_tr_res/repeats
    overall_performance_tr_res_std = performance_tr_res_std/repeats
    overall_performance_te_res = performance_te_res/repeats
    overall_performance_te_res_std = performance_te_res_std/repeats
    overall_performance_tr_rnn = performance_tr_rnn/repeats
    overall_performance_tr_rnn_std = performance_tr_rnn_std/repeats
    overall_performance_te_rnn = performance_te_rnn/repeats
    overall_performance_te_rnn_std = performance_te_rnn_std/repeats

    scores_with_sd = [(overall_performance_tr_res, overall_performance_tr_res_std, overall_performance_te_res, overall_performance_te_res_std),
                      (overall_performance_tr_rnn, overall_performance_tr_rnn_std, overall_performance_te_rnn, overall_performance_te_rnn_std),
                      (overall_performance_tr_ts, overall_performance_tr_ts_std, overall_performance_te_ts, overall_performance_te_ts_std)]
    print scores_with_sd
    util.print_table_row_performances_regression(feature_names[i], len(selected_train_X.index), len(selected_test_X.index), scores_with_sd)
    scores_over_all_algs.append(scores_with_sd)

DataViz.plot_performances_regression(['Reservoir', 'RNN', 'Time series'], feature_names, scores_over_all_algs)
'''
regr_train_y, regr_test_y = learner.reservoir_computing(train_X[features_after_chapter_5], train_y, test_X[features_after_chapter_5], test_y, gridsearch=True)
print regr_train_y, regr_test_y
DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['acc_phone_x'], test_X.index, test_y, regr_test_y['acc_phone_x'], 'acc_phone_x')
regr_train_y, regr_test_y = learner.recurrent_neural_network(train_X[basic_features], train_y, test_X[basic_features], test_y, gridsearch=True)
DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['acc_phone_x'], test_X.index, test_y, regr_test_y['acc_phone_x'], 'acc_phone_x')
regr_train_y, regr_test_y = learner.time_series(train_X[basic_features], train_y, test_X[features_after_chapter_5], test_y, gridsearch=True)
DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['acc_phone_x'], test_X.index, test_y, regr_test_y['acc_phone_x'], 'acc_phone_x')
