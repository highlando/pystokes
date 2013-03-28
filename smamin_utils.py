import numpy as np
import scipy.sparse as sps

def revert_sort_tob2(v,ColInd):
	"""revert to rearrangement of v used to make B2 

	invertible"""
	vra = np.zeros((len(v),1))
	ColIndC = np.setdiff1d(range(len(v)),ColInd)
	vra[ColInd,] = v[len(ColIndC):,]
	vra[ColIndC,] = v[:len(ColIndC),]

	return vra

def col_columns_atend(SparMat, ColInd):
	"""Shifts a set of columns of a sparse matrix to the right end.
	
	It takes a sparse matrix and a vector containing indices
	of columns which are appended at the right end 
	of the matrix while the remaining columns are shifted to left.

	"""
	
	mat_csr = sps.csr_matrix(SparMat, copy=True)
	MatWid = mat_csr.shape[1]

	# ColInd should not be altered
	ColIndC = np.copy(ColInd)

	for i in range(len(ColInd)):
		subind = ColIndC[i]

		# filter out the current column
		idx   = np.where(mat_csr.indices == subind)

		# shift all columns that were on the right by one to the left
		idxp  = np.where(mat_csr.indices >= subind)
		mat_csr.indices[idxp] -= 1

		# and adjust the ColInds for the replacement
		idsp = np.where(ColIndC >= subind)
		ColIndC[idsp] -= 1

		# append THE column at the end
		mat_csr.indices[idx] = MatWid - 1
		
	return mat_csr
