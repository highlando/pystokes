from time_int_schemes import col_columns_atend 
import numpy as np
import scipy.sparse as sps

N = 5 
n = 2
col = np.arange(0,N,n)
v = np.arange(N)
mat = sps.spdiags(v,[0],N,N+len(col))

MatRa = col_columns_atend(mat,col)
vra = np.append(v,col)
va  = np.append(v,0*col)
vra[col] = 0


