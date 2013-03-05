import unittest
import numpy as np
import scipy.sparse as sps

class TestSequenceFunctions(unittest.TestCase):

	def test_collectB2(self):
		from time_int_schemes import col_columns_atend
		
		N = 100
		n = 7
		col = np.arange(0,N,n)
		v = np.arange(N)
		mat = sps.spdiags(v,[0],N,N+len(col))

		MatRa = col_columns_atend(mat,col)
		vra = np.append(v,col)
		va  = np.append(v,0*col)
		vra[col] = 0

		
		self.assertTrue(np.allclose(MatRa*vra,mat*va))

if __name__ == '__main__':
    unittest.main()

