import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from Chapter7.LearningAlgorithms import ClassificationAlgorithms

data = pd.read_table("../datasets/our_data/plrx.txt")
algs = ClassificationAlgorithms()
data['label'] -= 1
train = data.head(130)
test = data.tail(52)
train_X = train[[str(i) for i in range(1,13)]]
train_y = train['label']
test_X = test[[str(i) for i in range(1,13)]]
test_y = test['label']
res = algs.feedforward_neural_network(train_X, train_y, test_X)	
def roc_curve_creation(P):
	roc_curve_points = []
	P = sorted(P, key = lambda x : x[0], reverse = True)
	pos = 0
	for pair in P:
		pos += pair[1]
	neg = len(P) - pos
	last = -1
	for i in range(len(P)):
		if P[i][0] == last:
			continue
		last = P[i][0]
		P_sel = P[i:]
		count = 0
		for pair in P_sel:
			count += pair[1]
		roc_curve_points.append([(len(P_sel) - count)/(1.0*neg), count/(1.0*pos)])
	return roc_curve_points

P = np.zeros((2,len(res[0])))
P[0] = res[2][1]
P[1] = train_y
P = P.transpose()

roc = roc_curve_creation(P)
roc = np.asmatrix(roc).transpose()
plot.plot(roc[0], roc[1], 'r+')
plot.show()
