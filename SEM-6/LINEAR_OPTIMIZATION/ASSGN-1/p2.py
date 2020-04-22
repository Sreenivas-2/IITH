import numpy as np
import sympy as sp
import math
from numpy.random import randint
import random

A = np.array([[1,1,0],
			 [0,-1,1],
			 [-1,0,0],
			 [0,-1,0],
			 [0,0,-1]])

b = np.array([1,0,0,0,0])

c = np.array([1,1,1])

# A = np.array([[1,1,2],
# 	 [-2,-2,-10],
# 	 [-1,0,0],
# 	 [0,-1,0],
# 	 [0,0,-1],
# 	 [0,0,1]])

# b = np.array([2,-10,0,0,0,1])

# c = np.array([0,0,1])

# A = np.array([[1,1],
# 	 [-2,-2],
# 	 [-1,0],
# 	 [0,-1]])

# b = np.array([2,-10,0,0])

# c = np.array([3,-1])


def initExtreme(A, b, c):

	b = b.reshape(-1, 1)
	A_ = np.hstack((A, b))
	add = np.zeros((2, A_.shape[1]))
	add[0][-1] = -1
	add[1][-1] = 1
	A_ = np.vstack((A_, add))
	b_ = np.vstack((b, np.array([0, 1]).reshape(-1,1)))
	c_ = np.zeros(A_.shape[1])
	c_[-1] = 1

	x = np.zeros(A_.shape[1])
	x[-1] = 1

	J = np.arange(A_.shape[1])
	while A_.shape[0] - 2 in J or np.linalg.det(A_[J,:]) == 0:
		J = np.array(randint(0, A_.shape[0] - 1, A_.shape[1]))
		# J = np.array(random.sample(range(0, A_.shape[0] - 1), 3))

	x = simplex(A_, b_, c_, x, J, False)
	if x[-1] > 0:
		return None, None

	J = getActiveSet(A, b, x[:-1])

	return x[:-1], J


def getActiveSet(A, b, x):
	'''
		A => m*n matrix where each row consists of coefficients of the constraints
		b => vector of length m, R.H.S values of the constraints
		x => extreme point
		returns => Active set at extreme point x
	'''

	M = (A.dot(x) - b).round(4)
	arr = np.array(np.where(M == 0)[0])
	return arr




def computeZ(A, c, J, flag):
	'''
		A => m*n matrix where each row consists of coefficients of the constraints
		c => coefficients in the objective function
		J => active set at extreme point
		returns => z
	'''

	Aj = A[J, :]
	c = c.reshape(-1,1)
	z = np.linalg.inv(Aj.T).dot(-c)

	if flag:
		if (z <= 0).all():
			return -1
		else:
			l = np.where(z>0)[0]
			index = np.where(J == np.min(J[l]))
			return index[0]
	else:
		if (z >= 0).all():
			return -1
		else:
			l = np.where(z<0)[0]
			index = np.where(J == np.min(J[l]))
			return index[0]


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

	k = A.dot(deltaX).round(5)
	index = np.array(np.where(k > 0)[0])
	A = A[index, :]
	b = b[index]
	x = x.reshape(-1,1)
	b = b.reshape(-1,1)
	a = b - A.dot(x)
	m = A.dot(deltaX)
	m.reshape(-1,1)
	arr = np.divide(a, m)
	alpha = np.min(arr)
	ind = np.argmin(arr)
	return alpha, index[ind]


def simplex (A, b, c, x, J, flag = True):

	k = computeZ(A, c, J, flag)

	cnt = 1
	while k != -1:
		print('Iteration - ' + str(cnt))
		cnt += 1
		deltaX = computeDeltaX(A, k, J)
		deltaX = deltaX.reshape(-1,1)

		if (A.dot(deltaX) <= 0).all():
			return None

		alpha, index = getAlpha(A, b, x, deltaX, J)
		l = alpha*deltaX
		x = x.reshape(-1,1)
		x = x + l
		J = np.delete(J, k)
		J = np.append(J, index)
		k = computeZ(A, c, J, flag)

	return x.round(5)

if __name__ == '__main__':

	x, J = initExtreme(A, b, c)
	if x is None:
		print('Infeasible')
	else:

		# Active set obtained here may lead to zero determinant (Check)
		if len(J) > A.shape[1]:
			J = J[:A.shape[1]]

		# x = x.round(5)
		x = simplex(A, b, c, x, J)
		if x is None:
			print('Unbounded')
		else:
			print(x.round(5))


		


