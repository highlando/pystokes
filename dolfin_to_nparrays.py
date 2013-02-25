from dolfin import *
import numpy as np
import scipy.sparse as sps

parameters.linear_algebra_backend = "uBLAS"

def get_sysNSmats( V, Q): # , velbcs ):
	""" Assembles the system matrices for Stokes equation

	in mixed FEM formulation, namely
		
		[ A  B' ] as [ Aa   Grada ] : W -> W'
		[ B  0  ]    [ Diva   0   ]
		
	for a given trial and test space W = V * Q and boundary conds.
	
	Plus the velocity mass matrix M.
	"""

	u = TrialFunction(V)
	p = TrialFunction(Q)
	v = TestFunction(V)
	q = TestFunction(Q)

	ma = inner(u,v)*dx
	aa = inner(grad(u), grad(v))*dx 
	grada = div(v)*p*dx
	diva = q*div(u)*dx

	# Assemble system
	M = assemble(ma)
	A = assemble(aa)
	Grad = assemble(grada)
	Div = assemble(diva)

	# Convert DOLFIN representation to numpy arrays
	rows, cols, values = M.data()
	Ma = sps.csr_matrix((values, cols, rows))

	rows, cols, values = A.data()
	Aa = sps.csr_matrix((values, cols, rows))

	rows, cols, values = Grad.data()
	BTa = sps.csr_matrix((values, cols, rows))

	rows, cols, values = Div.data()
	Ba = sps.csr_matrix((values, cols, rows))

	return Ma, Aa, BTa, Ba
	

def setget_rhs(V, Q, velbcs=None, t=None):

	if t is None:
		fv = Expression(("4*(x[0]*x[0]*x[0]*(6-12*x[1])+pow(x[0],4)*(6*x[1]-3)+x[1]*(1-3*x[1]+2*x[1]*x[1])"\
				"-6*x[0]*x[1]*(1-3*x[1]+2*x[1]*x[1])+3*x[0]*x[0]*(-1+4*x[1]-6*x[1]*x[1]+4*pow(x[1],3)))"\
				"+x[1]*(1-x[1])*(1-2*x[0])","-4*(- 3*(-1+x[1])*(-1+x[1])*x[1]*x[1]-3*x[0]*x[0]*(1-6*x[1]+6*x[1]*x[1])"\
				"+2*x[0]*x[0]*x[0]*(1-6*x[1]+6*x[1]*x[1])+x[0]*(1-6*x[1]+12*x[1]*x[1]-12*x[1]*x[1]*x[1]+6*x[1]*x[1]*x[1]*x[1]))"\
				"+ x[0]*(1-x[0])*(1-2*x[1])"))
	
	#fv = Expression(("sin(x[0])","0"))
	fp = Constant((0))

	v = TestFunction(V)
	q = TestFunction(Q)

	fv = inner(fv,v)*dx 
	fp = inner(fp,q)*dx

	fv = assemble(fv)
	fp = assemble(fp)

	fv = fv.array()
	fv = fv.reshape(len(fv), 1)

	fp = fp.array()
	fp = fp.reshape(len(fp), 1)

	return fv, fp

def get_convvec(u0, V):
	"""return the convection vector e.g. for explicit schemes
	"""

	v = TestFunction(V)
	ConvForm = inner(grad(u0)*u0, v)*dx

	ConvForm = assemble(ConvForm)
	ConvVec = ConvForm.array()
	ConvVec = ConvVec.reshape(len(ConvVec), 1)

	return ConvVec

def condense_sysmatsbybcs(Ma,Aa,BTa,Ba,fv,fp,velbcs):
	"""resolve the Dirichlet BCs and condense the sysmats

	to the inner nodes"""

	auxu = np.zeros((len(fv),1))
	bcinds = []
	for bc in velbcs:
		bcdict = bc.get_boundary_values()
		auxu[bcdict.keys(),0] = bcdict.values()
		bcinds.extend(bcdict.keys())

	# accumulating the bcs to the right hand sides
	fv = fv - Aa*auxu    # '*' is np.dot for csr matrices
	fp = fp - Ba*auxu
	
	# indices of the innernodes
	innerinds = np.setdiff1d(range(len(fv)),bcinds)

	# extract the inner nodes equation coefficients
	Mc = Ma[innerinds,:][:,innerinds]
	Ac = Aa[innerinds,:][:,innerinds]
	fvc= fv[innerinds,:]
	Bc  = Ba[:,innerinds]
	BTc = BTa[innerinds,:]

	bcvals = auxu[bcinds]

	# removal of the indefiniteness in pressure via pi_0 !=! 0
	# eeeh, better not here :/
	# Bc  = Ba[1:,innerinds]
	# BTc = BTa[innerinds,1:]
	# fp  = fp[1:,0]

	return Mc, Ac, BTc, Bc, fvc, fp, bcinds, bcvals, innerinds
