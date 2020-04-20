import numpy as np
import sympy as sp
import math
from numpy.linalg import matrix_rank

A = np.array([[-1,0,0],
	 [0,-1,0],
	 [0,0,-1],
	 [1,0,0],
	 [0,1,0],
	 [0,0,1],
	 [1,1,1]])

b = np.array([0,0,0,2,2,2,5])

c = np.array([1, 1, -1])

# A = np.array([[1,-2,-2,3],
# 	 [2,-3,-1,1],
# 	 [0,0,1,0],
# 	 [-1,0,0,0],
# 	 [0,-1,0,0],
# 	 [0,0,-1,0],
# 	 [0,0,0,-1]], dtype = np.float64)

# b = np.array([0,0,1,0,0,0,0], dtype = np.float64)

# c = np.array([-3,5,-1,2], dtype = np.float64)

# A = np.array([[5,4],
# 	 [1,2]])

# b = np.array([32,10])

# c = np.array([-2,-3])


def initExtreme(A, b):
	# b_ = b
	# b = b.reshape(-1,1)
	# M = np.hstack((A, b))
	# _, inds = sp.Matrix(M).T.rref()

	# if len(inds) > A.shape[1]:
	# 	inds = inds[:A.shape[1]]

	# inds = np.array(inds)
	# print(inds)
	# A = A[inds, :]
	# b_ = b_[inds]
	# b_ = b_.reshape(-1,1)
	# x = np.linalg.inv(A).dot(b_)
	# return x, inds
	return np.array([2,2,0]), np.array([2,3,4])
	# return np.array([0,0,2]), np.array([0,1,5])
	# return np.array([0,0,0,0]), np.array([3,4,5,6])


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
	# print('z',z)
	if (z >= 0).all():
		return -1
	else:
		# print('---',np.where(z<0))
		l = np.where(z<0)[0]
		# print(l)
		index = np.where( J == np.min(J[l]))
		# index = np.min(np.where(z < 0))
		# J[index]
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
	# print(A)
	x = x.reshape(-1,1)
	# print(x)
	# print(b)
	b = b.reshape(-1,1)
	a = b - A.dot(x)
	m = A.dot(deltaX)
	m.reshape(-1,1)
	# print('a',a)
	# print('m',m)
	arr = np.divide(a, m)
	# print(arr)
	alpha = np.min(arr)
	ind = np.argmin(arr)
	return alpha, index[ind]


def simplex (A, b, c):
	x, inds = initExtreme(A, b)
	# print(x)
	# J = getActiveSet(A, b, x)
	J = inds
	k = computeZ(A, c, J)
	x_ = np.zeros(x.shape)
	# print('1\n-', J + 1)
	cnt = 2
	while k != -1:
		print(cnt)
		# if cnt == 8:
			# break
		cnt += 1
		deltaX = computeDeltaX(A, k, J)
		deltaX = np.asarray(deltaX, dtype = np.float64)
		alpha, index = getAlpha(A, b, x, deltaX, J)
		l = alpha*deltaX
		# print('k',k)
		# print(A.dot(deltaX))
		# print('deltaX', deltaX)
		# print('alpha', alpha)
		x = x.reshape(-1,1)
		x = x + l
		J = np.delete(J, k)
		J = np.append(J, index)
		# print('-',J + 1)
		k = computeZ(A, c, J)

	print(x)

if __name__ == '__main__':
	simplex(A, b, c)
		


