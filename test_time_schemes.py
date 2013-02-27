from dolfin import *
import numpy as np
import scipy.sparse as sps

import dolfin_to_nparrays as dtn
reload(dtn)
import time_int_schemes as tis
reload(tis)
		
import smartminex_tayhoomesh 

class TimestepParams(object):
	def __init__(self, method):
		self.t0 = 0
		self.tE = 0.1
		self.Nts = 10
		self.method = method
		self.UpFiles = UpFiles(method)

def solve_stokesTimeDep():
	"""system to solve
	
  	 	 du\dt - lap u + grad p = fv
		         div u          = fp
	
	"""

	N = 32 
	method = 2


	methdict = {0:'ImpEulFull', 
			1:'ImpEulQr', 
			2:'HalfExpEulInd2'}

	print 'You have chosen %s for time integration' % methdict[method]

	# instantiate object containing mesh, V, Q, velbcs, invinds
	PrP = ProbParams(N)

	# instantiate the Time Int Parameters
	TsP = TimestepParams(methdict[method])

	# get system matrices as np.arrays
	Ma, Aa, BTa, Ba = dtn.get_sysNSmats(PrP.V, PrP.Q)
	fv, fp = dtn.setget_rhs(PrP.V, PrP.Q, PrP.fv, PrP.fp)

	Mc, Ac, BTc, Bc, fvc, fp, bcinds, bcvals, invinds = \
			dtn.condense_sysmatsbybcs(Ma,Aa,BTa,Ba,fv,fp,PrP.velbcs)
	
	###
	# Time stepping
	###
	# starting value
	dimredsys = len(fvc)+len(fp)-1
	vp_init   = np.zeros((dimredsys,1))
	
	if method == 0:
		tis.plain_impeuler(Mc,Ac,BTc,Bc,fvc,fp,vp_init, 
				PrP,TsP=TsP)
	elif method == 1:
		tis.qr_impeuler(Mc,Ac,BTc,Bc,fvc,fp,vp_init,PrP,TsP=TsP)
	elif method == 2:
		tis.halfexp_euler_nseind2(Mc,Ac,BTc,Bc,fvc,fp,vp_init,PrP,TsP=TsP)
	
	#vp_stat = np.linalg.solve(sadSysmat[:-1,:-1],np.vstack([fvc,fp[:-1],]))
	#v, p = expand_vp_dolfunc(invinds,velbcs,V,Q,
	#		vp=vp_stat,vc=None,pc=None)
	#u_file << v, 1
	#p_file << p, 1

	return 

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

class ProbParams(object):
	def __init__(self,N=None):
		if N is not None:
			self.mesh = smartminex_tayhoomesh.getmake_mesh(N)
		else:
			self.mesh = smartminex_tayhoomesh.getmake_mesh(16)

		self.V = VectorFunctionSpace(self.mesh, "CG", 2)
		self.Q = FunctionSpace(self.mesh, "CG", 1)
		self.velbcs = setget_velbcs_zerosq(self.mesh, self.V)
		self.fp = Constant((0))
		self.fv = Expression(("4*(x[0]*x[0]*x[0]*(6-12*x[1])+pow(x[0],4)*(6*x[1]-3)+x[1]*(1-3*x[1]+2*x[1]*x[1])"\
				"-6*x[0]*x[1]*(1-3*x[1]+2*x[1]*x[1])+3*x[0]*x[0]*(-1+4*x[1]-6*x[1]*x[1]+4*pow(x[1],3)))"\
				"+x[1]*(1-x[1])*(1-2*x[0])","-4*(- 3*(-1+x[1])*(-1+x[1])*x[1]*x[1]-3*x[0]*x[0]*(1-6*x[1]+6*x[1]*x[1])"\
				"+2*x[0]*x[0]*x[0]*(1-6*x[1]+6*x[1]*x[1])+x[0]*(1-6*x[1]+12*x[1]*x[1]-12*x[1]*x[1]*x[1]+6*x[1]*x[1]*x[1]*x[1]))"\
				"+ x[0]*(1-x[0])*(1-2*x[1])"))

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
