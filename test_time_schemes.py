from dolfin import *
import numpy as np
import scipy.sparse as sps
from scipy.linalg import qr
import smartminex_tayhoomesh 

parameters.linear_algebra_backend = "uBLAS"

def solve_stokesTimeDep(debu=None):
	"""system to solve
	
  	 	 du\dt - lap u + grad p = fv
		         div u          = fp

	"""

	N = 8

	# instantiate object containing mesh, V, Q, velbcs, invinds
	PrP = ProbParams(N)

	# get system matrices as np.arrays
	Ma, Aa, BTa, Ba = get_sysNSmats(PrP.V, PrP.Q)
	fv, fp = setget_rhs(PrP.V, PrP.Q)

	Mc, Ac, BTc, Bc, fvc, fp, bcinds, bcvals, invinds = \
			condense_sysmatsbybcs(Ma,Aa,BTa,Ba,fv,fp,PrP.velbcs)
	
	###
	# Time stepping
	###
	# starting value
	dimredsys = len(fvc)+len(fp)-1
	vp_init   = np.zeros((dimredsys,1))

	qr_impeuler(Mc,Ac,BTc,Bc,fvc,fp,vp_init,PrP,TsP=None)
	#plain_impeuler(Mc,Ac,BTc,Bc,fvc,fp,vp_init,PrP,TsP=None)
	
	#vp_stat = np.linalg.solve(sadSysmat[:-1,:-1],np.vstack([fvc,fp[:-1],]))
	#v, p = expand_vp_dolfunc(invinds,velbcs,V,Q,
	#		vp=vp_stat,vc=None,pc=None)
	#u_file << v, 1
	#p_file << p, 1

	return 

def qr_impeuler(Mc,Ac,BTc,Bc,fvc,fp,vp_init,PrP,TsP=None):
	""" with BTc[:-1,:] = Q*[R ; 0] 
	we define ~M = Q*M*Q.T , ~A = Q*A*Q.T , ~V = Q*v
	and condense the system accordingly """

	Nts, t0, tE = 10, 0, 1.
	dt = (t0-tE)/Nts

	Nv = len(fvc)
	Np = len(fp)

	upf = UpFiles('QrImpEul')

	v, p   = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None)

	tcur = t0

	upf.u_file << v, tcur
	upf.p_file << p, tcur

	# Use SVD, as QR does not give the full Q matrix
	# QR = BT is like QR^2QT = BT*B
	Qm, Rm = qr(BTc.todense()[:,:-1],mode='full')
	#Rm = np.diag(np.sqrt(Sm[:Np-1]))

	Qm_1 = Qm[:Np-1,]   #first Np-1 rows of Q
	Qm_2 = Qm[Np-1:,]   #last Nv-(Np-1) rows of Q

	TM_11 = np.dot(Qm_1,np.dot(Mc.todense(),Qm_1.T))
	TA_11 = np.dot(Qm_1,np.dot(Ac.todense(),Qm_1.T))

	TM_21 = np.dot(Qm_2,np.dot(Mc.todense(),Qm_1.T))
	TA_21 = np.dot(Qm_2,np.dot(Ac.todense(),Qm_1.T))

	TM_12 = np.dot(Qm_1,np.dot(Mc.todense(),Qm_2.T))
	TA_12 = np.dot(Qm_1,np.dot(Ac.todense(),Qm_2.T))

	TM_22 = np.dot(Qm_2,np.dot(Mc.todense(),Qm_2.T))
	TA_22 = np.dot(Qm_2,np.dot(Ac.todense(),Qm_2.T))

	Tv1 = np.linalg.solve(Rm[:Np-1,].T, fp[:-1,])

	Tv2_old = np.dot(Qm_2, vp_init[:Nv,])

	Tfv2 = np.dot(Qm_2, fvc) + np.dot(TA_21, Tv1)

	IterA2  = TM_22-dt*TA_22

	for i in range(Nts):
		tcur = tcur + dt

		Iter2rhs = np.dot(TM_22, Tv2_old) + dt*Tfv2
		Tv2_new = np.linalg.solve(IterA2, Iter2rhs)

		Tv2_old = Tv2_new

		# Retransformation v = Q.T*Tv
		vc_new = np.dot(Qm.T, np.vstack([Tv1, Tv2_new]))
		
		RhsTv2dot = Tfv2 + np.dot(TA_22, Tv2_new) 
		Tv2dot = np.linalg.solve(TM_22, RhsTv2dot)

		RhsP = np.dot(Qm_1,fvc) + np.dot(TA_11,Tv1) \
				+ np.dot(TA_12,Tv2_new) - np.dot(TM_12,Tv2dot)

		pc_new = np.linalg.solve(Rm[:Np-1,],RhsP)

		v, p = expand_vp_dolfunc(PrP, vp=None, vc=vc_new, pc=pc_new)
		upf.u_file << v, tcur
		upf.p_file << p, tcur
		
	return

def plain_impeuler(Mc,Ac,BTc,Bc,fvc,fp,vp_init,PrP,TsP=None):

	Nts, t0, tE = 10, 0, 1.
	dt = (t0-tE)/Nts

	Nv = len(fvc)
	Np = len(fp)

	upf = UpFiles('ImpEul')

	v, p   = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None)

	tcur = t0

	upf.u_file << v, tcur
	upf.p_file << p, tcur

	IterAv = sps.hstack([Mc-dt*Ac,dt*BTc])
	IterAp = sps.hstack([dt*Bc,sps.csr_matrix((Np,Np))])
	IterA  = sps.vstack([IterAv,IterAp]).todense()[:-1,:-1]

	vp_old = vp_init
	for i in range(Nts):
		tcur = tcur + dt
		#iterateeee
		Iterrhs = np.vstack([Mc*vp_old[:Nv,],np.zeros((Np-1,1))]) \
				+ dt*np.vstack([fvc,fp[:-1,]])
		vp_new = np.linalg.solve(IterA,Iterrhs)
		v, p = expand_vp_dolfunc(PrP, vp=vp_new, vc=None, pc=None)
		upf.u_file << v, tcur
		upf.p_file << p, tcur
		vp_old = vp_new
		
	return

def expand_vp_dolfunc(PrP, vp=None, vc=None, pc=None):
	"""expand v and p to the dolfin function representation"""

	v = Function(PrP.V)
	p = Function(PrP.Q)

	if vp is not None:
		vc = vp[:len(PrP.invinds),:]
		pc = vp[len(PrP.invinds):,:]

	ve = np.zeros((PrP.V.dim(),1))

	# fill in the boundary values
	for bc in PrP.velbcs:
		bcdict = bc.get_boundary_values()
		ve[bcdict.keys(),0] = bcdict.values()

	ve[PrP.invinds] = vc

	pe = np.vstack([pc,[0]])

	v.vector().set_local(ve)
	p.vector().set_local(pe)

	return v, p

def setget_velbcs_zerosq(mesh, V):
	# Boundaries
	def top(x, on_boundary): 
		return  np.fabs(x[1] - 1.0 ) < DOLFIN_EPS 
			  # and (np.fabs(x[0]) > DOLFIN_EPS))
			  # and np.fabs(x[0] - 1.0) > DOLFIN_EPS )
			  

	def leftbotright(x, on_boundary): 
		return ( np.fabs(x[0] - 1.0) < DOLFIN_EPS 
				or np.fabs(x[1]) < DOLFIN_EPS 
				or np.fabs(x[0]) < DOLFIN_EPS)

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

class TimestepParams(object):
	def __init__(self):
		self.t0 = 0
		self.tE = 1
		self.Nts = 10


class ProbParams(object):
	def __init__(self,N=None):
		if N is not None:
			self.mesh = smartminex_tayhoomesh.getmake_mesh(N)
		else:
			self.mesh = smartminex_tayhoomesh.getmake_mesh(16)

		self.V = VectorFunctionSpace(self.mesh, "CG", 2)
		self.Q = FunctionSpace(self.mesh, "CG", 1)
		self.velbcs = setget_velbcs_zerosq(self.mesh, self.V)

		bcinds = []
		for bc in self.velbcs:
			bcdict = bc.get_boundary_values()
			bcinds.extend(bcdict.keys())

		# indices of the inner velocity nodes
		self.invinds = np.setdiff1d(range(self.V.dim()),bcinds)

class UpFiles(object):
	def __init__(self, name=None):
		if name is not None:
			self.u_file = File("results/%s_velocity.pvd" % name)
			self.p_file = File("results/%s_pressure.pvd" % name)
		else:
			self.u_file = File("results/velocity.pvd")
			self.p_file = File("results/pressure.pvd")

if __name__ == '__main__':
	solve_stokesTimeDep()
