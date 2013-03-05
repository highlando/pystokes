import scipy.sparse 
import numpy as np

# test matrix
mat = np.zeros((5,5))
mat[[1, 2, 3, 3], [0, 2, 2, 4]] = 1
mat = scipy.sparse.lil_matrix(mat)

# which columns to collect at the end of the matrix
sub = np.array([0,2])

def col_columns_atend(SparMat,ColInd):
	"""get a sparse matrix and a vector containing indices

	of columns that are appended at the right end 
	of the matrix. The remaining columns are shifted to left.
	"""
	
	mat_csr = scipy.sparse.csr_matrix(SparMat)
	MatWid = mat_csr.shape[1]

	# ColInd should not be altered
	ColIndC = np.copy(ColInd)

	for i in range(len(ColInd)):
		subind = ColInd[i]
		idx   = np.where(mat_csr.indices == subind)
		# shift all columns of higher index by one to the left
		idxp  = np.where(mat_csr.indices >= subind)
		mat_csr.indices[idxp] -= 1
		# and adjust the ColInds for the replacement
		idsp = np.where(ColInd >= subind)
		sub[idsp] -= 1

		# append THE column at the end
		mat_csr.indices[idx] = MatWid - 1
		
		return mat_csr

print mat_csr.todense()
