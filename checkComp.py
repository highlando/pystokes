from dolfin import *
from scipy.sparse import csr_matrix
import numpy as np

parameters.linear_algebra_backend = "uBLAS"

mixed = False

mesh = UnitSquare(2, 2)

if mixed is True:
	# Define mixed FEM function spaces
	V = VectorFunctionSpace(mesh, "CG", 2)
	Q = FunctionSpace(mesh, "CG", 1)
	W = V * Q

	(u, p) = TrialFunctions(W)
	(v, q) = TestFunctions(W)

else:
	# Define FEM function spaces
	V = VectorFunctionSpace(mesh, "CG", 2)

	u = TrialFunction(V)
	v = TestFunction(V)

# define Form
aa = inner(grad(u), grad(v))*dx 

# Assemble system
A = assemble(aa)

# righthandside
fvhomo = Constant((1,2))
Lvh = inner(fvhomo,v)*dx 
b = assemble(Lvh)

## Convert DOLFIN representation to numpy arrays
rows, cols, values = A.data()
Aa = csr_matrix((values, cols, rows))
ba = b.array()
ba = ba.reshape(len(ba), 1)

print ba.shape
	
