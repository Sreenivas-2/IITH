'''
  Ashwanth Kumar : CS17BTECH11017
  Sai Sreenivas  : CS17BTECH11012
  Sasikanth B    : CS17BTECH11010
'''

import numpy as np
import random

def RemoveDegenerancy(A, b):
	'''
		Removes Degeneracy by adding small values (epsilons) to our original vector b

		A => m*n matrix where each row consists of coefficients of the constraints
		b => vector of length m, R.H.S values of the constraints
		returns => modified vector b
	'''

	indices = []
	for index, row in enumerate(b):
		if row == 0:
			arr = A[index]
			if len(np.where(arr == 0.)[0]) == A.shape[1] - 1 and (arr[np.nonzero(arr)] == 1 or arr[np.nonzero(arr)] == -1):
				indices.append(index)

	int_list = random.sample(range(1, 100), A.shape[0])
	float_list = np.array(sorted([x/1000000 for x in int_list], key = np.float64)).reshape(-1,1)
	float_list[indices,:] = 0
	float_list[-1] = 0

	b = b + np.array(float_list).reshape(-1,1)
	
	return b

def initExtreme(A, b):
	'''
		A => m*n matrix where each row consists of coefficients of the constraints
		b => vector of length m, R.H.S values of the constraints
		returns => Initial extreme point and the active set (indices of n linearly independent rows of A)
	'''

	c_ = np.zeros(A.shape[1] + 1)
	c_[-1] = 1
	x = np.zeros(A.shape[1] + 1)
	x[-1] = np.min(b) - 1

	indices = []
	for index, row in enumerate(b):
		if row == 0:
			arr = A[index]
			if len(np.where(arr == 0.)[0]) == A.shape[1] - 1 and (arr[np.nonzero(arr)] == 1 or arr[np.nonzero(arr)] == -1):
				indices.append(index)

	ones = np.ones((A.shape[0], 1))
	ones[np.array(indices), -1] = 0
	A_ = np.hstack((A, ones))
	add = np.zeros((2, A_.shape[1]))
	add[0][-1] = 1
	add[1][-1] = -1

	A_ = np.vstack((A_, add))
	b_ = np.vstack((b, np.array([0, -x[-1]]).reshape(-1,1)))

	indices.append(A_.shape[0] - 1)
	J = np.array(indices)

	x, J = simplex(A_, b_, c_, x, J)

	if x[-1] < 0:
		return None, None

	J = get_ActiveSet(A, b, x[:-1])

	return x[:-1], J


def get_ActiveSet(A, b, x):
	'''
		A => m*n matrix where each row consists of coefficients of the constraints
		b => vector of length m, R.H.S values of the constraints
		x => current extreme point
		returns => Active set at extreme point x (indices of n linearly independent rows of A)
	'''

	x = x.reshape(-1,1)
	M = A.dot(x) - b
	arr = np.array(np.where((M >= -0.000000001) & (M <= 0.000000001))[0])
	return arr




def get_InactiveIndex(A, c, J):
	'''
		A => m*n matrix where each row consists of coefficients of the constraints
		c => coefficients in the objective function
		J => active set at current extreme point
		returns => Index of the row present in the active set which shall be removed for obtaining next neighboring point.
	'''

	Aj = A[J, :]
	c = c.reshape(-1,1)
	z = np.linalg.inv(Aj.T).dot(-c)

	if (z <= 0).all():
		return -1
	else:
		l = np.where(z>0)[0]
		index = np.where(J == np.max(J[l]))
		return index[0]


def get_DirVec(A, k, J):
	'''
		A => m*n matrix where each row consists of coefficients of the constraints
		k => index of element with z < 0
		J => active set at current extreme point
		returns => direction vector from current point to the neighboring point
	'''

	Aj = A[J, :]
	rhs = np.zeros((Aj.shape[1],1))
	rhs[k] = -1
	DirVec = np.linalg.inv(Aj).dot(rhs)
	return DirVec


def get_Factor_and_Index(A, b, x, DirVec, J):
	'''
		x' = x + (alpha)*(DirVec)

		A => m*n matrix where each row consists of coefficients of the constraints
		b => vector of length m, R.H.S values of the constraints
		x => current extreme point
		DirVec => direction vector from current point to the neighboring point
		J => active set at the current extreme point
		returns => alpha, index (index of new row which shall replace index obtainded from get_InactiveIndex function in the active set)
	'''

	k = A.dot(DirVec)
	index = np.array(np.where(k > 0)[0])
	index = np.array(list(set(list(index)) - set(list(J))))
	A = A[index, :]
	b = b[index]
	x = x.reshape(-1,1)
	b = b.reshape(-1,1)
	a = b - A.dot(x)
	m = A.dot(DirVec)
	m.reshape(-1,1)
	arr = np.divide(a, m)
	alpha = np.min(arr)
	ind = np.argmin(arr)
	return alpha, index[ind]


def simplex(A, b, c, x, J):
	'''
		Main Algorithm Implementation

		A => m*n matrix where each row consists of coefficients of the constraints
		b => vector of length m, R.H.S values of the constraints
		c => coefficients in the objective function
		x => Initial extreme point
		J => active set at the current extreme point
		returns => x (solution), J (final active set)
	'''

	k = get_InactiveIndex(A, c, J)
	# cnt = 1
	while k != -1:
		# print('Iteration - ' + str(cnt))
		# cnt += 1

		DirVec = get_DirVec(A, k, J)
		DirVec = DirVec.reshape(-1,1)
		if (A.dot(DirVec) <= 0).all():
			return None, None

		alpha, index = get_Factor_and_Index(A, b, x, DirVec, J)
		x = x.reshape(-1,1)
		x = x + (alpha * DirVec)
		J = np.delete(J, k)
		J = np.append(J, index)
		k = get_InactiveIndex(A, c, J)

	return x, J


def main(A, b, c):

	b = b.reshape(-1, 1)
	if (b >= 0).all():
		x = np.zeros((1, A.shape[1]))
		J = get_ActiveSet(A, b, x)
		if len(J) > A.shape[1]:
			return main(A, RemoveDegenerancy(A, b), c)

	else:
		x, J = initExtreme(A, b)
		if x is not None and len(J) > A.shape[1]:
			return main(A, RemoveDegenerancy(A, b), c)

	if x is None:
		return None, -1

	else:
		c = c.reshape(-1, 1)
		x = x.reshape(-1, 1)
		x, J = simplex(A, b, c, x, J)
		if x is None:
			return None, -2

		else:
			return x, J


if __name__ == '__main__':


	# Input Section
	rows = int(input('Enter the number of constraints (number of rows in A) : '))
	print('Enter the elements of the matrix A row wise with row elements space separated (same as a matrix form):')

	A = []
	for i in range(rows):
	    a = list(map(float, input().split()))
	    A.append(a)
	A = np.array(A)

	print('Enter the elements of b space separated:')
	b = list(map(float, input().split()))
	b = np.array(b)

	print('Enter the elements of c space separated:')
	c = list(map(float, input().split()))
	c = np.array(c)

	# Inputs Given
	print('\nInputs Given:')
	print('A\n', A)
	print('b\n', b)
	print('c\n', c)
	print()

	# Algorithm
	x, J = main(A, b, c)
	if x is not None:
		x = x.round(3)
		print('Solution :\n', x)
		print('Maximum Value of c :', c.dot(x.reshape(-1,1)))

	elif J == -1:
		print('Infeasible Solution')

	else:
		print('Unbounded Solution')


		


