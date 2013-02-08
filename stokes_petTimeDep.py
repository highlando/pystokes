from dolfin import *
from scipy.sparse import csr_matrix
import numpy as np

parameters.linear_algebra_backend = "uBLAS"

def solve_stokesTimeDep(debu=None):

	# get the mesh 
	N = 2
	# mesh = UnitSquare(N, N)
	mesh = getmake_mesh_smaminext(N)

	# Define mixed FEM function spaces
	V = VectorFunctionSpace(mesh, "CG", 2)
	Q = FunctionSpace(mesh, "CG", 1)
	velbcs = setget_velbcs_zerosq(mesh, V)

	# get system matrices as np.arrays
	Aa, BTa, Ba = get_sysNSmats(V, Q)
	
	fv, fp = setget_rhs(V, Q, velbcs)

	if debu is not None:
		return Aa, BTa, Ba, velbcs, fv, fp, mesh, V, Q
	else:
		return

def get_ij_subgrid(k,N):
	"""to get i,j numbering of the cluster centers of smaminext"""

	n = N-1
	if k > n**2-1 or k < 0:
		raise Exception('%s: No such node on the subgrid!' % k)
		
	j = np.mod(k,n)
	i = (k-j)/n
	return j, i

def getmake_mesh_smaminext(N):
	"""write the mesh for the smart minext tayHood square

	order is I. main grid, II. subgrid = grid of the cluster centers
	and in I and II lexikographical order
	first y-dir, then x-dir """

	try:
		f = open('smegrid%s.xml' % N)
	except IOError:
		print 'Need generate the mesh...'

		# main grid
		h = 1./(N-1)
		y, x = np.ogrid[0:N,0:N]
		y = h*y+0*x
		x = h*x+0*y
		mgrid = np.hstack((y.reshape(N**2,1), x.reshape(N**2,1)))

		# sub grid
		y, x = np.ogrid[0:N-1,0:N-1]
		y = h*y+0*x
		x = h*x+0*y
		sgrid = np.hstack((y.reshape((N-1)**2,1), x.reshape((N-1)**2,1)))

		grid = np.vstack((mgrid,sgrid+0.5*h))

		f = open('smegrid%s.xml' % N, 'w')
		f.write('<?xml version="1.0"?> \n <dolfin xmlns:dolfin="http://www.fenicsproject.org"> \n <mesh celltype="triangle" dim="2"> \n')

		f.write('<vertices size="%s">\n' % (N**2+(N-1)**2) )
		for k in range(N**2+(N-1)**2):
			f.write('<vertex index="%s" x="%s" y="%s" />\n' % (k, grid[k,0], grid[k,1]))
		
		f.write('</vertices>\n')
		f.write('<cells size="%s">\n' % (4*(N-1)**2))
		for j in range(N-1):
			for i in range(N-1):
				# number of current cluster center
				k = j*(N-1) + i 
				# vertices of the main grid in the cluster
				v0, v1, v2, v3 = j*N+i, (j+1)*N+i, (j+1)*N+i+1, j*N+i+1 

				f.write('<triangle index="%s" v0="%s" v1="%s" v2="%s" />\n' % (4*k,   v0, N**2+k, v1))
				f.write('<triangle index="%s" v0="%s" v1="%s" v2="%s" />\n' % (4*k+1, v1, N**2+k, v2))
				f.write('<triangle index="%s" v0="%s" v1="%s" v2="%s" />\n' % (4*k+2, v2, N**2+k, v3)) 
				f.write('<triangle index="%s" v0="%s" v1="%s" v2="%s" />\n' % (4*k+3, v3, N**2+k, v0)) 

		f.write('</cells>\n')
		
		f.write('</mesh> \n </dolfin> \n')
		f.close()

		print 'done'

	mesh = Mesh('smegrid%s.xml' % N)

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

	return Aa, BTa, Ba
	
def setget_rhs(V, Q, velbcs, t=None):

	if t is not None:
	#	fv = Expression(("4*(x[0]*x[0]*x[0]*(6-12*x[1])+pow(x[0],4)*(6*x[1]-3)+x[1]*(1-3*x[1]+2*x[1]*x[1])"\
	#			"-6*x[0]*x[1]*(1-3*x[1]+2*x[1]*x[1])+3*x[0]*x[0]*(-1+4*x[1]-6*x[1]*x[1]+4*pow(x[1],3)))"\
	#			"+x[1]*(1-x[1])*(1-2*x[0])","-4*(- 3*(-1+x[1])*(-1+x[1])*x[1]*x[1]-3*x[0]*x[0]*(1-6*x[1]+6*x[1]*x[1])"\
	#			"+2*x[0]*x[0]*x[0]*(1-6*x[1]+6*x[1]*x[1])+x[0]*(1-6*x[1]+12*x[1]*x[1]-12*x[1]*x[1]*x[1]+6*x[1]*x[1]*x[1]*x[1]))"\
	#			"+ x[0]*(1-x[0])*(1-2*x[1])"))
		print 'haha'
	
	

	fv = Expression(("sin(x[0])","0"))
	fp = Constant((0))

	v = TestFunction(V)
	q = TestFunction(Q)

	fv = inner(fv,v)*dx 
	fp = inner(fp,q)*dx

	fv = assemble(fv)
	fp = assemble(fp)

	for bc in velbcs:
		bc.apply(fv)

	fv = fv.array()
	fv = fv.reshape(len(fv), 1)

	fp = fp.array()
	fp = fp.reshape(len(fp), 1)

	return fv, fp

	#for bc in velbcs:
	#	bc.apply(A, fv)

def condense_sysmatsbybcs(Aa=None,BTa=None,Ba=None,
		fv=None,fp=None,velbcs=None):
	"""resolve the Dirichlet BCs and condense the sysmats

	to the inner nodes"""

	return



if __name__ == '__main__':
	solve_stokesTimeDep()

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


    def test_sub_dofmap(self):
        """ Test extraction of sub- and sub-sub-dofmaps."""
        mesh = UnitSquare(1, 1)
        V = FunctionSpace(mesh, "CG", 1)
        Q = VectorFunctionSpace(mesh, "CG", 1)
        W = V * Q

        for cell in cells(mesh):
            dofs0 = numpy.array((0,)*3, dtype="I")
            dofs1 = numpy.array((0,)*3, dtype="I")
            dofs2 = numpy.array((0,)*3, dtype="I")
            dofs3 = numpy.array((0,)*6, dtype="I")

            W.sub(0).dofmap().tabulate_dofs(dofs0, cell)

            L = W.sub(1)
            L.sub(0).dofmap().tabulate_dofs(dofs1, cell)
            L.sub(1).dofmap().tabulate_dofs(dofs2, cell)
            L.dofmap().tabulate_dofs(dofs3, cell)

            self.assertEqual(len(numpy.intersect1d(dofs0, dofs1)), 0)
            self.assertEqual(len(numpy.intersect1d(dofs0, dofs2)), 0)
            self.assertEqual(len(numpy.intersect1d(dofs1, dofs2)), 0)
            self.assertTrue(numpy.array_equal(numpy.append(dofs1, dofs2), dofs3))
            self.assertTrue(numpy.array_equal(numpy.append(dofs1, dofs2), dofs3))
