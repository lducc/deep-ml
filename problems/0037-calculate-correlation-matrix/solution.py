import numpy as np

def calculate_correlation_matrix(X, Y=None):
	# Your code here
	if Y is None:
		return np.corrcoef(X, rowvar=False)
	else:
		corr = np.corrcoef(X, Y, rowvar=False)
		n = X.shape[1]
		return corr[:n, n:]