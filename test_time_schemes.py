from dolfin import *
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import pickle

import dolfin_to_nparrays as dtn
reload(dtn)
import time_int_schemes as tis
reload(tis)
		
import smartminex_tayhoomesh 
reload(smartminex_tayhoomesh)
class TimestepParams(object):
	def __init__(self, method, N):
		self.t0 = 0
		self.tE = 4.0
		self.Nts = 64
		self.Ntslist = [8, 16, 32, 64]#, 128] #, 256]#, 512]#, 1024]
		self.method = method
		self.UpFiles = UpFiles(method)
		self.Residuals = NseResiduals()
		self.linatol = 1e-6 #1e-8   # 0 for direct sparse solver
		self.PickleFile = 'pickles/NTs%dto%dMesh%d' % (self.Ntslist[0], self.Ntslist[-1], N) + method

def solve_stokesTimeDep():
	"""system to solve
	
  	 	 du\dt - lap u + grad p = fv
		         div u          = fp
	
	"""

	N = 20
	method = 2

	methdict = {0:'ImpEulFull', 
			1:'ImpEulQr', 
			2:'HalfExpEulInd2',
			3:'HalfExpEulSmaMin',
			4:'HalfExpEulSmaMinSplit'}

	# instantiate object containing mesh, V, Q, velbcs, invinds
	PrP = ProbParams(N)

	# instantiate the Time Int Parameters
	TsP = TimestepParams(methdict[method], N)

	print 'Mesh parameter N = %d' % N
	print 'You have chosen %s for time integration' % methdict[method]
	print 'The tolerance for the linear solver is %e' %TsP.linatol

	#prepare for pickling the residuals
	fpic = file(TsP.PickleFile,'w')
	pickle.dump(TsP.Residuals, fpic)

	# get system matrices as np.arrays
	Ma, Aa, BTa, Ba = dtn.get_sysNSmats(PrP.V, PrP.Q)
	fv, fp = dtn.setget_rhs(PrP.V, PrP.Q, PrP.fv, PrP.fp)

	Mc, Ac, BTc, Bc, fvbc, fpbc, bcinds, bcvals, invinds = \
			dtn.condense_sysmatsbybcs(Ma,Aa,BTa,Ba,fv,fp,PrP.velbcs)
	
	###
	# Time stepping
	###
	# starting value
	dimredsys = len(fvbc)+len(fp)-1
	vp_init   = np.zeros((dimredsys,1))
	
	for i, CurNTs in enumerate(TsP.Ntslist):
		TsP.Nts = CurNTs

		if method == 0:
			tis.plain_impeuler(Mc,Ac,BTc,Bc,fvbc,fp,vp_init, 
					PrP,TsP=TsP)
		elif method == 1:
			tis.qr_impeuler(Mc,Ac,BTc,Bc,fvbc,fp,vp_init,PrP,TsP=TsP)
		elif method == 2:
			tis.halfexp_euler_nseind2(Mc,Ac,BTc,Bc,fvbc,fpbc,vp_init,PrP,TsP=TsP)
		else:
			# get the indices of the bubbles of B2
			# the 1st pressure dof is the one that is removed
			B2BubInds = smartminex_tayhoomesh.get_B2_bubbleinds(N, PrP.V, PrP.mesh)
			# we need the B2Bub indices in the reduced setting vc
			# this gives a masked array of boolean type
			B2BubBool = np.in1d(np.arange(PrP.V.dim())[PrP.invinds], B2BubInds)
			#B2BubInds = np.arange(len(B2BubIndsBool

			if method == 3:
				tis.halfexp_euler_smarminex(Mc,Ac,BTc,Bc,fvbc,fpbc,vp_init,B2BubBool,PrP,TsP=TsP)
			elif method == 4:
				tis.halfexp_euler_smarminex_split(Mc,Ac,BTc,Bc,fvbc,fpbc,vp_init,B2BubBool,PrP,TsP=TsP)
	
	plot_errs_res(TsP)
		
	#vp_stat = np.linalg.solve(sadSysmat[:-1,:-1],np.vstack([fvc,fp[:-1],]))
	#v, p = expand_vp_dolfunc(invinds,velbcs,V,Q,
	#		vp=vp_stat,vc=None,pc=None)
	# u_file << v, 1
	#p_file << p, 1

	return TsP, PrP

def save_simu(TsP,PrP):

	return

def plot_errs_res(TsP):

	plt.close('all')
	for i in range(len(TsP.Ntslist)):
		fig1 = plt.figure(1)
		plt.plot(TsP.Residuals.ContiRes[i])
		plt.title('Lina residual in the continuity eqn')
		fig2 = plt.figure(2)
		plt.plot(TsP.Residuals.VelEr[i])
		plt.title('Error in the velocity')
		fig3 = plt.figure(3)
		plt.plot(TsP.Residuals.PEr[i])
		plt.title('Error in the pressure')

	plt.show(block=False)

	return


def plot_exactsolution(PrP,TsP):

	u_file = File("results/exa_velocity.pvd")
	p_file = File("results/exa_pressure.pvd")
	for tcur in np.linspace(TsP.t0,TsP.tE,11):
		PrP.v.t = tcur
		PrP.p.t = tcur
		vcur = project(PrP.v,PrP.V)
		pcur = project(PrP.p,PrP.Q)
		u_file << vcur, tcur
		p_file << pcur, tcur


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
		self.nu = 0
		self.fp = Constant((0))
		self.fv = Expression(("40*nu*pow(x[0],2)*pow(x[1],3)*sin(t) - 60*nu*pow(x[0],2)*pow(x[1],2)*sin(t) + 24*nu*pow(x[0],2)*x[1]*pow((x[0] - 1),2)*sin(t) + 20*nu*pow(x[0],2)*x[1]*sin(t) - 12*nu*pow(x[0],2)*pow((x[0] - 1),2)*sin(t) - 32*nu*x[0]*pow(x[1],3)*sin(t) + 48*nu*x[0]*pow(x[1],2)*sin(t) - 16*nu*x[0]*x[1]*sin(t) + 8*nu*pow(x[1],3)*pow((x[0] - 1),2)*sin(t) - 12*nu*pow(x[1],2)*pow((x[0] - 1),2)*sin(t) + 4*nu*x[1]*pow((x[0] - 1),2)*sin(t) - 4*pow(x[0],3)*pow(x[1],2)*pow((x[0] - 1),3)*(2*x[0] - 1)*pow((x[1] - 1),2)*(2*x[1]*(x[1] - 1) + x[1]*(2*x[1] - 1) + (x[1] - 1)*(2*x[1] - 1) - 2*pow((2*x[1] - 1),2))*pow(sin(t),2) - 4*pow(x[0],2)*pow(x[1],3)*pow((x[0] - 1),2)*cos(t) + 6*pow(x[0],2)*pow(x[1],2)*pow((x[0] - 1),2)*cos(t) - 2*pow(x[0],2)*x[1]*pow((x[0] - 1),2)*cos(t) + 2*x[0]*pow(x[1],2)*sin(t) - 2*x[0]*x[1]*sin(t) - pow(x[1],2)*sin(t) + x[1]*sin(t)", "-40*nu*pow(x[0],3)*pow(x[1],2)*sin(t) + 32*nu*pow(x[0],3)*x[1]*sin(t) - 8*nu*pow(x[0],3)*pow((x[1] - 1),2)*sin(t) + 60*nu*pow(x[0],2)*pow(x[1],2)*sin(t) - 48*nu*pow(x[0],2)*x[1]*sin(t) + 12*nu*pow(x[0],2)*pow((x[1] - 1),2)*sin(t) - 24*nu*x[0]*pow(x[1],2)*pow((x[1] - 1),2)*sin(t) - 20*nu*x[0]*pow(x[1],2)*sin(t) + 16*nu*x[0]*x[1]*sin(t) - 4*nu*x[0]*pow((x[1] - 1),2)*sin(t) + 12*nu*pow(x[1],2)*pow((x[1] - 1),2)*sin(t) + 4*pow(x[0],3)*pow(x[1],2)*pow((x[1] - 1),2)*cos(t) - 4*pow(x[0],2)*pow(x[1],3)*pow((x[0] - 1),2)*pow((x[1] - 1),3)*(2*x[1] - 1)*(2*x[0]*(x[0] - 1) + x[0]*(2*x[0] - 1) + (x[0] - 1)*(2*x[0] - 1) - 2*pow((2*x[0] - 1),2))*pow(sin(t),2) - 6*pow(x[0],2)*pow(x[1],2)*pow((x[1] - 1),2)*cos(t) + 2*pow(x[0],2)*x[1]*sin(t) - pow(x[0],2)*sin(t) + 2*x[0]*pow(x[1],2)*pow((x[1] - 1),2)*cos(t) - 2*x[0]*x[1]*sin(t) + x[0]*sin(t)"), t=0, nu=self.nu)

		self.v = Expression((
			"sin(t)*x[0]*x[0]*(1 - x[0])*(1 - x[0])*2*x[1]*(1 - x[1])*(2*x[1] - 1)", 
			"sin(t)*x[1]*x[1]*(1 - x[1])*(1 - x[1])*2*x[0]*(1 - x[0])*(1 - 2*x[0])"), t = 0)
		self.vdot = Expression((
			"cos(t)*x[0]*x[0]*(1 - x[0])*(1 - x[0])*2*x[1]*(1 - x[1])*(2*x[1] - 1)",
			"cos(t)*x[1]*x[1]*(1 - x[1])*(1 - x[1])*2*x[0]*(1 - x[0])*(1 - 2*x[0])"), t = 0)
		self.p =  Expression(("sin(t)*x[0]*(1-x[0])*x[1]*(1-x[1])"), t = 0)

		bcinds = []
		for bc in self.velbcs:
			bcdict = bc.get_boundary_values()
			bcinds.extend(bcdict.keys())

		# indices of the inner velocity nodes
		self.invinds = np.setdiff1d(range(self.V.dim()),bcinds)

class NseResiduals(object):
	def __init__(self):
		self.ContiRes = []
		self.VelEr = []
		self.PEr = []

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
