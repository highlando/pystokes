from dolfin import *
from scipy.sparse import csr_matrix
import numpy as np

parameters.linear_algebra_backend = "uBLAS"

mesh = UnitSquare(4, 4)

plot(mesh)


# Define mixed FEM function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

def top(x, on_boundary): 
  return x[1] > 1.0 - DOLFIN_EPS 

def leftbotright(x, on_boundary): 
  return ( x[0] > 1.0 - DOLFIN_EPS 
			or x[1] < DOLFIN_EPS 
			or x[0] < DOLFIN_EPS)

# No-slip boundary condition for velocity
noslip = Constant((0.0, 0.0))
bc0 = DirichletBC(W.sub(0), noslip, leftbotright)

# Boundary condition for velocity at the lid
lid = Constant(("0.0", "0.0"))
bc1 = DirichletBC(W.sub(0), lid, top)

# Collect boundary conditions
bcs = [bc0, bc1]

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

aa = inner(grad(u), grad(v))*dx 
grada = div(v)*p*dx
diva = q*div(u)*dx

# Assemble system
A = assemble(aa)
Grad = assemble(grada)
Div = assemble(diva)

fvhomo = Constant((1,2))
fphomo = Constant((0))

Lvh = inner(fvhomo,v)*dx 
Lph = inner(fphomo,q)*dx

b = assemble(Lvh)

for bc in bcs:
	bc.apply(A)
	bc.apply(b)


# Convert DOLFIN representation to numpy arrays
rows, cols, values = A.data()
Aa = csr_matrix((values, cols, rows))
ba = b.array()
ba = ba.reshape(len(ba), 1)
#	
## get system matrices as np.arrays
#Aa, grada, diva = get_sysNSMats(W,bcs)
#
