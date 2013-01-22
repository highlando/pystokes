from dolfin import *

def main():
	# Test for PETSc or Epetra
	if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Epetra"):
		info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
		exit()

	parameters.linear_algebra_backend = 'uBLAS'

	# Load mesh
	mesh = UnitSquareMesh(8, 8)
	plot(mesh)

	# Define function spaces
	V = VectorFunctionSpace(mesh, "CG", 2)
	Q = FunctionSpace(mesh, "CG", 1)
	W = V * Q

	# Boundaries
	#def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
	#def left(x, on_boundary): return x[0] < DOLFIN_EPS
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
	lid = Constant(("1", "0.0"))
	bc1 = DirichletBC(W.sub(0), lid, top)

	# Collect boundary conditions
	bcs = [bc0, bc1]

def getCurStokesMat():
	main()
	# Define variational problem
	(u, p) = TrialFunctions(W)
	(v, q) = TestFunctions(W)
	f = Constant((0.0, 0.0))
	a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
	L = inner(f, v)*dx

	# Assemble system
	A, bb = assemble_system(a, L, bcs)

	return A, bb

if __name__ == '__main__':
    main()
