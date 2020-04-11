import numpy as np
import math

A = np.array([[4,2],
	 [1,2],
	 [1,1]])

B = np.array([16,8,5])

C = [3, 2]


def construct_table (C, A, B):
	'''
		Constructs a table for representation of the dictionary
		C => coefficients in the objective function
		A => m*n matrix where each row consists of coefficients of the constraints
		B => vector of length m, R.H.S values of the constraints
		returns => zip as shown below
				   	|C 0|
				   	|A B|
	'''
	
	obj_ = np.hstack((C, [0]))
	B = B.reshape(-1,1)
	constraints_ = np.hstack((A, B))
	table = np.vstack((obj_, constraints_))
	table = np.asarray(table, dtype = np.float64)
	return table


def feasible (M):
	'''
		Checks if the dictionary is feasible or not
		M => coefficients in the objective function
		reutrns => boolean, if feasible true else false
	'''
	
	return any(coeff > 0 for coeff in M[:-1])


def getPivot (M):
	'''
		M => table (representaion of the dictionary)
		returns => corresponding index of the row and column of the pivot
	'''

	column = np.argmax(M[0][:-1])
	list_ = []

	for index, line in enumerate(M[1:]):
		coeff = line[column]
		list_.append(math.inf if coeff <= 0 else line[-1]/coeff)

	row = list_.index(min(list_)) + 1

	return (row, column)


def updateTable (M, pivotPos):
	'''
		M => table (representaion of the dictionary)
		pivotPos => (i, j) in table for the pivot of dictionary
		returns => updated table
	'''

	newTable = np.zeros(M.shape)

	row, column = pivotPos
	pivot = M[row][column]
	newTable[row] = M[row]/pivot

	for i in range(M.shape[0]):
		if i != row:
			factor = newTable[row] * M[i][column]
			newTable[i] = M[i] - factor

	return newTable


def simplex (C, A, B):
	'''
		C => coefficients in the objective function
		A => m*n matrix where each row consists of coefficients of the constraints
		B => vector of length m, R.H.S values of the constraints
		returns => solutions for the dictionary
	'''

	table = construct_table(C, A, B)

	while feasible(table[0]):
		pivotPos = getPivot(table)
		table = updateTable(table, pivotPos)

	return -table[0][-1]


if __name__ == '__main__':

	print(simplex(C, A, B))
