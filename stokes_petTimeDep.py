from dolfin import *
from scipy.sparse import csr_matrix
import numpy as np

parameters.linear_algebra_backend = "uBLAS"

def get_mesh_usq():
	# Load mesh
	return mesh = UnitSquareMesh(16, 16)

def set_bcs_zerosq(mesh,W):
	# Boundaries
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

	return bcs

def get_stoksysmats( W, bcs ):
	""" Assembles the system matrices for Stokes equation

	in mixed FEM formulation, namely
		
		[ A  B' ] as [ Aa   Grada ] : W -> W'
		[ B  0  ]    [ Diva   0   ]
		
	for a given trial and test space W = V * Q and boundary conds.
	"""

	(u, p) = TrialFunctions(W)
	(v, q) = TestFunctions(W)

	aa = inner(grad(u), grad(v))*dx 
	grada = div(v)*p*dx
	diva = q*div(u)*dx

	# Assemble system
	A = assemble_system(aa)
	Grad = assemble_system(grada)
	Div = assemble_system(diva)

	fvhomo = constant((0,0))
	fphomo = constant((0))

	Lvh = inner(fvhomo,v)*dx 
	Lph = inner(fphomo,q)*dx

	for bc in bcs:
		bc.apply(A, Lvh)


	TODO: bcs hier????

	# Convert DOLFIN representation to numpy arrays
	rows, cols, values = A.data()
	Aa = csr_matrix((values, cols, rows))
	ba = b.array()
	ba = ba.reshape(len(ba), 1)
	
fv = Expression(("4*(x[0]*x[0]*x[0]*(6-12*x[1])+pow(x[0],4)*(6*x[1]-3)+x[1]*(1-3*x[1]+2*x[1]*x[1])"\
		"-6*x[0]*x[1]*(1-3*x[1]+2*x[1]*x[1])+3*x[0]*x[0]*(-1+4*x[1]-6*x[1]*x[1]+4*pow(x[1],3)))"\
		"+x[1]*(1-x[1])*(1-2*x[0])","-4*(-3*(-1+x[1])*(-1+x[1])*x[1]*x[1]-3*x[0]*x[0]*(1-6*x[1]+6*x[1]*x[1])"\
		"+2*x[0]*x[0]*x[0]*(1-6*x[1]+6*x[1]*x[1])+x[0]*(1-6*x[1]+12*x[1]*x[1]-12*x[1]*x[1]*x[1]+6*x[1]*x[1]*x[1]*x[1]))"\
		"+ x[0]*(1-x[0])*(1-2*x[1])"))

	L = inner(fv, v)*dx + inner(fp,q)*dx
fp = Constant((0))



def solve_stokesTimeDep():

	# get the mesh 
	mesh = get_mesh_usq()

	# Define mixed FEM function spaces
	V = VectorFunctionSpace(mesh, "CG", 2)
	Q = FunctionSpace(mesh, "CG", 1)
	W = V * Q

	bcs = set_bcs_zerosq(mesh,W)

	# get system matrices as np.arrays
	Aa, grada, diva = get_sysNSMats(W,bcs)


# Create Krylov solver and preconditioner
solver = KrylovSolver("gmres", "none")
solver.set_operator(A)

# Solve
U = Function(W)
solver.solve(U.vector(), b)

# Get sub-functions
u, p = U.split()

# Save solution in VTK format
ufile_pvd = File("velocity.pvd")
ufile_pvd << u
pfile_pvd = File("pressure.pvd")
pfile_pvd << p

# Plot solution
plot(u)
plot(p)
interactive()
