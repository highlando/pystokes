from dolfin import *
from scipy.sparse import csr_matrix
import numpy as np

parameters.linear_algebra_backend = "uBLAS"

def solve_stokesTimeDep():

	# get the mesh 
	mesh = get_mesh_usq(4)

	# Define mixed FEM function spaces
	V = VectorFunctionSpace(mesh, "CG", 2)
	Q = FunctionSpace(mesh, "CG", 1)

	velbcs = setget_velbcs_zerosq(mesh, V)

	# get system matrices as np.arrays
	Aa, BTa, Ba = get_sysNSmats(V, Q)
	
	fv, fp = setget_rhs(V, Q) #, velbcs)

## Create Krylov solver and preconditioner
#solver = KrylovSolver("gmres", "none")
#solver.set_operator(A)
#
## Solve
#U = Function(W)
#solver.solve(U.vector(), b)
#
## Get sub-functions
#u, p = U.split()
#
## Save solution in VTK format
#ufile_pvd = File("velocity.pvd")
#ufile_pvd << u
#pfile_pvd = File("pressure.pvd")
#pfile_pvd << p
#
## Plot solution
#plot(u)
#plot(p)
#interactive()

def get_mesh_usq(ndof):
	# Load mesh
	mesh = UnitSquareMesh(ndof, ndof)
	return mesh

def setget_velbcs_zerosq(mesh, V):
	# Boundaries
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
	lid = Constant((0.0, 0.0))
	bc1 = DirichletBC(V, lid, top)

	# Collect boundary conditions
	velbcs = [bc0, bc1]

	return velbcs

def get_sysNSmats( V, Q): # , velbcs ):
	""" Assembles the system matrices for Stokes equation

	in mixed FEM formulation, namely
		
		[ A  B' ] as [ Aa   Grada ] : W -> W'
		[ B  0  ]    [ Diva   0   ]
		
	for a given trial and test space W = V * Q and boundary conds.
	"""

	u = TrialFunction(V)
	p = TrialFunction(Q)
	v = TestFunction(V)
	q = TestFunction(Q)

	aa = inner(grad(u), grad(v))*dx 
	grada = div(v)*p*dx
	diva = q*div(u)*dx

	# Assemble system
	A = assemble(aa)
	Grad = assemble(grada)
	Div = assemble(diva)

	# Convert DOLFIN representation to numpy arrays
	rows, cols, values = A.data()
	Aa = csr_matrix((values, cols, rows))

	rows, cols, values = Grad.data()
	BTa = csr_matrix((values, cols, rows))

	rows, cols, values = Div.data()
	Ba = csr_matrix((values, cols, rows))

	rhs = fv.array()
	rhs = rhs.reshape(len(rhs), 1)

	return Aa, BTa, Ba, rhs
	
def setget_rhs(V, Q, velbcs, t=None):

	if t is None:
		fv = Expression(("4*(x[0]*x[0]*x[0]*(6-12*x[1])+pow(x[0],4)*(6*x[1]-3)+x[1]*(1-3*x[1]+2*x[1]*x[1])"\
				"-6*x[0]*x[1]*(1-3*x[1]+2*x[1]*x[1])+3*x[0]*x[0]*(-1+4*x[1]-6*x[1]*x[1]+4*pow(x[1],3)))"\
				"+x[1]*(1-x[1])*(1-2*x[0])","-4*(-3*(-1+x[1])*(-1+x[1])*x[1]*x[1]-3*x[0]*x[0]*(1-6*x[1]+6*x[1]*x[1])"\
				"+2*x[0]*x[0]*x[0]*(1-6*x[1]+6*x[1]*x[1])+x[0]*(1-6*x[1]+12*x[1]*x[1]-12*x[1]*x[1]*x[1]+6*x[1]*x[1]*x[1]*x[1]))"\
				"+ x[0]*(1-x[0])*(1-2*x[1])"))

	fp = Constant((0))

	v = TestFunction(V)
	q = TestFunction(Q)

	fv = inner(fv,v)*dx 
	fp = inner(fp,q)*dx

	fv = assemble(fv)
	fp = assemble(fp)

	for bc in velbcs f**k f**k  :
		bc.apply(fv)

	fv = fv.array()
	fv = fv.reshape(len(fv), 1)

	fp = fp.array()
	fp = fp.reshape(len(fp), 1)

	return fv, fp

	#for bc in velbcs:
	#	bc.apply(A, fv)

def condense_sysmatsbybcs(Aa=None,BTa=None,Ba=None,
		fv=None,fp=None,velbcs):
	"""


if __name__ == '__main__':
	solve_stokesTimeDep()
