import numpy as np
from scipy import linalg

def initialize_echo_state_network(inputs, outputs, reservoir):
	Win = (np.random.rand(reservoir,1+inputs)-0.5) * 1
	W = np.random.rand(reservoir,reservoir)-0.5
	for i in range(reservoir):
		dontkeep = np.random.choice(reservoir,size=3*reservoir/4, replace = False)
		for j in dontkeep:
			W[i,j] = 0
	Wback = (np.random.rand(reservoir,outputs)-0.5) * 1

	# Adjust W to "guarantee" the echo state property.
	rhoW = max(abs(linalg.eig(W)[0]))
	W *= 1.25 / rhoW
	return Win, W, Wback

def initialize_echo_state_network1(inputs, outputs, reservoir):
	Win = (np.random.rand(reservoir,1+inputs)-0.5) * 1
	W = np.random.standard_cauchy(reservoir,reservoir)
	for i in range(reservoir):
		dontkeep = np.random.choice(reservoir,size=3*reservoir/4, replace = False)
		for j in dontkeep:
			W[i,j] = 0
	Wback = (np.random.rand(reservoir,outputs)-0.5) * 1

	# Adjust W to "guarantee" the echo state property.
	rhoW = max(abs(linalg.eig(W)[0]))
	W *= 1.25 / rhoW
	return Win, W, Wback
