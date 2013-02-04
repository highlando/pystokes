from dolfin import *
from scipy.sparse import csr_matrix
import numpy as np

parameters.linear_algebra_backend = "uBLAS"

mesh = UnitSquareMesh(3, 3)


# Define mixed FEM function spaces
V = VectorFunctionSpace(mesh, "CG", 2)

def top(x, on_boundary): 
  return x[1] > 1.0 - DOLFIN_EPS 

def leftbotright(x, on_boundary): 
  return ( x[0] > 1.0 - DOLFIN_EPS 
			or x[1] < DOLFIN_EPS 
			or x[0] < DOLFIN_EPS)

# No-slip boundary condition for velocity
noslip = Constant((0.0, 0.0))
bc0 = DirichletBC(V, noslip, leftbotright)

# Boundary condition for velocity at the lid
lid = Constant(("0.0", "0.0"))
bc1 = DirichletBC(V, lid, top)

# Collect boundary conditions
bcs = [bc0, bc1]

u = TrialFunction(V)
v = TestFunction(V)

ga = inner(grad(u), grad(v))*dx 

# Assemble system
A = assemble(ga)

fvhomo = Constant((0,0))

Lvh = inner(fvhomo,v)*dx 
b = assemble(Lvh)

for bc in bcs:
	bc.zero(A)
	bc.apply(b)

## Convert DOLFIN representation to numpy arrays
rows, cols, values = A.data()
Aa = csr_matrix((values, cols, rows))
ba = b.array()
ba = ba.reshape(len(ba), 1)
	
# get system matrices as np.arrays
# Aa = get_sysNSMats(W,bcs)


