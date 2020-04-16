import numpy as np
import math

A = np.array([[-1,0,0],
	 [0,-1,0],
	 [0,0,-1],
	 [1,0,0],
	 [0,1,0],
	 [0,0,1],
	 [1,1,1]])

b = np.array([0,0,0,2,2,2,5])

c = np.array([1, 1, -1])


def initExtreme():
	return np.array([2,2,0])


def getActiveSet(A, b, x):
	'''
		A => m*n matrix where each row consists of coefficients of the constraints
		b => vector of length m, R.H.S values of the constraints
		x => extreme point
		returns => Active set at extreme point x
	'''

	M = A.dot(x) - b
	arr = np.array(np.where(M == 0)[0])
	return arr




def computeZ(A, c, J):
	'''
		A => m*n matrix where each row consists of coefficients of the constraints
		c => coefficients in the objective function
		J => active set at extreme point
		returns => z
	'''

	Aj = A[J, :]
	z = np.linalg.inv(Aj.T).dot(-c)
	if (z >= 0).all():
		return -1
	else:
		index = np.min(np.where(z < 0))
		return index


def computeDeltaX(A, k, J):
	'''
		A => m*n matrix where each row consists of coefficients of the constraints
		k => indix of element with z < 0
		returns => deltaX
	'''

	Aj = A[J, :]
	rhs = np.zeros((Aj.shape[1],1))
	rhs[k] = -1
	deltaX = np.linalg.inv(Aj).dot(rhs)
	return deltaX


def getAlpha(A, b, x, deltaX, J):
	'''
		A => m*n matrix where each row consists of coefficients of the constraints
		b => vector of length m, R.H.S values of the constraints
		returns => alpha
	'''

	index = np.array(np.where(A.dot(deltaX) > 0)[0])
	A = A[index, :]
	b = b[index]
	b = b.reshape(-1,1)
	a = b - A.dot(x)
	m = A.dot(deltaX)
	m.reshape(-1,1)
	arr = np.divide(a, m)
	alpha = np.min(arr)
	ind = np.argmin(arr)
	return alpha, index[ind]


def simplex (A, b, c):
	x = initExtreme()
	J = getActiveSet(A, b, x)
	k = computeZ(A, c, J)
	x_ = np.zeros(x.shape)

	while k != -1:
		deltaX = computeDeltaX(A, k, J)
		alpha, index = getAlpha(A, b, x, deltaX, J)
		l = alpha*deltaX
		x = x.reshape(-1,1)
		x = x + l
		# print(x_)
		J = np.delete(J, k)
		J = np.append(J, index)
		k = computeZ(A, c, J)

	print(x)

if __name__ == '__main__':
	simplex(A, b, c)
		


