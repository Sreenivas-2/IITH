import numpy as np

A = np.array([[1,1],
	 [0,1],
	 [1,2]])

B = np.array([6,3,9])

C = [2, 5]


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

	return any(coeff > 0 for coeff in M)

def getPivot (M):
	'''
		M => table (representaion of the dictionary)
		returns => corresponding index of the row and column of the pivot
	'''

	



def simplex (C, A, B):
	'''
		C => coefficients in the objective function
		A => m*n matrix where each row consists of coefficients of the constraints
		B => vector of length m, R.H.S values of the constraints
		returns => solutions for the dictionary
	'''

	table = construct_table(C, A, B)

	# while feasible(table[0]):
	# 	break;

	return solution()

if __name__ == '__main__':

	construct_table(C, A, B)