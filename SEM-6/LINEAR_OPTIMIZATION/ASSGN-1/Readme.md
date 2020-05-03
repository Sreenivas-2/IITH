In this assignment we will solve the standard maximization problem using Simplex Method

(1) Problem - 1 : (Non Degenerate Case)

	Maximize c
	Ax <= b 

(2) Problem - 2 : (Degenerate Case)
	Maximize c
	Ax <= b


-- Input is accepted manually

-- Some test cases are attached (test_cases.txt)

(**) References :

	=> (http://www.seas.ucla.edu/~vandenbe/ee236a/lectures/simplex.pdf)

	=> For initial Point Generation :

		Case - 1 : If entries in b are >= 0 then we can have x with all 0s as initial feasible point

		Case - 2 : If b has atleast one negative entry then :

					We will introduce new variable z with condition z >= min(b) - 1 and z <= 0 with adding another column to original A with entries 0 for rows with expressing the sign of variables,

					else fill with 1, Now the initial extreme point will be x (all zeroes) and z (min(b) - 1) so that only n equations are active (i.e no degeneracy)

					example :

						    A         <=       b        

					     0  1  1 			  -2			     
					     0 -1  0			   0                 ) 
					    -1  0  0			   0				 ) => Modified matrix A after the above operations ( here 1,2,4 rows are active)
					     0  0 -1			   0				 )
					     0  0  1			  -3


	=> For degeneracy we follow the epsilon method mentioned here (Degeneracy section) : (https://www.cse.iitb.ac.in/~sundar/linear_optimization/lecture16.pdf)
