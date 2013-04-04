from dolfin import *

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import os, glob

import dolfin_to_nparrays as dtn
import time_int_schemes as tis
import smartminex_tayhoomesh 

from plot_utils import save_simu

class TimestepParams(object):
	def __init__(self, method, N):
		self.t0 = 0
		self.tE = None
		self.Omega = None
		self.Ntslist = [256, 512]#, 1024]#, 2048]
		self.NOutPutPts = 32
		self.method = method
		self.Split = None  #Can be 'Full' and 'Semi'
		self.SadPtPrec = True
		self.UpFiles = UpFiles(method)
		self.Residuals = NseResiduals()
		self.linatol = 1e-4 #1e-8   # 0 for direct sparse solver
		self.MaxIter = None
		self.Ml = None  #preconditioners
		self.Mr = None
		self.ParaviewOutput = False

def solve_stokesTimeDep(method=None, Omega=3, tE=1.0, Split=None, Prec=None, N=None, NtsList=None, LinaTol=None, MaxIter=None):
	"""system to solve
	
  	 	 du\dt - lap u + grad p = fv
		         div u          = fp
	
	"""

	if N is None:
		N = 20 

	if method is None:
		method = 2

	methdict = {0:'ImpEulFull', 
			1:'ImpEulQr', 
			2:'HalfExpEulInd2',
			3:'HalfExpEulSmaMin'}

	# instantiate object containing mesh, V, Q, velbcs, invinds
	PrP = ProbParams(N,Omega)
	# instantiate the Time Int Parameters
	TsP = TimestepParams(methdict[method], N)
	if NtsList is not None:
		TsP.Ntslist = NtsList
	if LinaTol is not None:
		TsP.linatol = LinaTol
	if Split is not None:
		TsP.Split = Split
	if MaxIter is not None:
		TsP.MaxIter = MaxIter
	
	TsP.tE = tE
	TsP.Omega = Omega

	print 'Mesh parameter N = %d' % N
	print 'Time interval [%d,%d]' % (TsP.t0, TsP.tE)
	print 'Omega = %d' % TsP.Omega
	print 'You have chosen %s for time integration' % methdict[method]
	if Split:
		print 'The system is split'
	print 'The tolerance for the linear solver is %e' %TsP.linatol

	# get system matrices as np.arrays
	Ma, Aa, BTa, Ba, MPa = dtn.get_sysNSmats(PrP.V, PrP.Q)
	fv, fp = dtn.setget_rhs(PrP.V, PrP.Q, PrP.fv, PrP.fp)

	# condense the system by resolving the boundary values
	Mc, Ac, BTc, Bc, fvbc, fpbc, bcinds, bcvals, invinds = \
			dtn.condense_sysmatsbybcs(Ma,Aa,BTa,Ba,fv,fp,PrP.velbcs)
	
	if method > 2:
		from smamin_utils import col_columns_atend

		MSmeCL, BSme, B2Inds, B2BoolInv, B2BI = smartminex_tayhoomesh.get_smamin_rearrangement(N,PrP,Mc,Bc)

		FvbcSme = np.vstack([fvbc[~B2BoolInv,],fvbc[B2BoolInv,]])
		FpbcSme = fpbc

		PrP.Pdof = 0 # Thats how the smamin is constructed
	

	### Output
	if TsP.ParaviewOutput :
		if not os.getcwd().split(os.sep)[-1] == 'pystokes':
			raise Warning('You are not in the right directory')
		
		os.chdir('results/')
		for fname in glob.glob(TsP.method + '*'):
			os.remove( fname )

		os.chdir('..')

	
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
		elif method == 3:
			tis.halfexp_euler_smarminex(MSmeCL,BSme,MPa,FvbcSme,FpbcSme,
					vp_init,B2BoolInv,PrP,TsP)

		# Output only in first iteration!
		TsP.ParaviewOutput = False
	
	save_simu(TsP, PrP)
	#plot_errs_res(TsP)
		
	#vp_stat = np.linalg.solve(sadSysmat[:-1,:-1],np.vstack([fvc,fp[:-1],]))
	#v, p = expand_vp_dolfunc(invinds,velbcs,V,Q,
	#		vp=vp_stat,vc=None,pc=None)
	# u_file << v, 1
	#p_file << p, 1

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
	def __init__(self,N,Omega):

		self.mesh = smartminex_tayhoomesh.getmake_mesh(N)
		self.N = N
		self.V = VectorFunctionSpace(self.mesh, "CG", 2)
		self.Q = FunctionSpace(self.mesh, "CG", 1)
		self.velbcs = setget_velbcs_zerosq(self.mesh, self.V)
		self.Pdof = 0  #dof removed in the p approximation
		self.omega = Omega
		self.nu = 0
		self.fp = Constant((0))
		self.fv = Expression(("40*nu*pow(x[0],2)*pow(x[1],3)*sin(omega*t) - 60*nu*pow(x[0],2)*pow(x[1],2)*sin(omega*t) + 24*nu*pow(x[0],2)*x[1]*pow((x[0] - 1),2)*sin(omega*t) + 20*nu*pow(x[0],2)*x[1]*sin(omega*t) - 12*nu*pow(x[0],2)*pow((x[0] - 1),2)*sin(omega*t) - 32*nu*x[0]*pow(x[1],3)*sin(omega*t) + 48*nu*x[0]*pow(x[1],2)*sin(omega*t) - 16*nu*x[0]*x[1]*sin(omega*t) + 8*nu*pow(x[1],3)*pow((x[0] - 1),2)*sin(omega*t) - 12*nu*pow(x[1],2)*pow((x[0] - 1),2)*sin(omega*t) + 4*nu*x[1]*pow((x[0] - 1),2)*sin(omega*t) - 4*pow(x[0],3)*pow(x[1],2)*pow((x[0] - 1),3)*(2*x[0] - 1)*pow((x[1] - 1),2)*(2*x[1]*(x[1] - 1) + x[1]*(2*x[1] - 1) + (x[1] - 1)*(2*x[1] - 1) - 2*pow((2*x[1] - 1),2))*pow(sin(omega*t),2) - 4*pow(x[0],2)*pow(x[1],3)*pow((x[0] - 1),2)*omega*cos(omega*t) + 6*pow(x[0],2)*pow(x[1],2)*pow((x[0] - 1),2)*omega*cos(omega*t) - 2*pow(x[0],2)*x[1]*pow((x[0] - 1),2)*omega*cos(omega*t) + 2*x[0]*pow(x[1],2)*sin(omega*t) - 2*x[0]*x[1]*sin(omega*t) - pow(x[1],2)*sin(omega*t) + x[1]*sin(omega*t)", "-40*nu*pow(x[0],3)*pow(x[1],2)*sin(omega*t) + 32*nu*pow(x[0],3)*x[1]*sin(omega*t) - 8*nu*pow(x[0],3)*pow((x[1] - 1),2)*sin(omega*t) + 60*nu*pow(x[0],2)*pow(x[1],2)*sin(omega*t) - 48*nu*pow(x[0],2)*x[1]*sin(omega*t) + 12*nu*pow(x[0],2)*pow((x[1] - 1),2)*sin(omega*t) - 24*nu*x[0]*pow(x[1],2)*pow((x[1] - 1),2)*sin(omega*t) - 20*nu*x[0]*pow(x[1],2)*sin(omega*t) + 16*nu*x[0]*x[1]*sin(omega*t) - 4*nu*x[0]*pow((x[1] - 1),2)*sin(omega*t) + 12*nu*pow(x[1],2)*pow((x[1] - 1),2)*sin(omega*t) + 4*pow(x[0],3)*pow(x[1],2)*pow((x[1] - 1),2)*omega*cos(omega*t) - 4*pow(x[0],2)*pow(x[1],3)*pow((x[0] - 1),2)*pow((x[1] - 1),3)*(2*x[1] - 1)*(2*x[0]*(x[0] - 1) + x[0]*(2*x[0] - 1) + (x[0] - 1)*(2*x[0] - 1) - 2*pow((2*x[0] - 1),2))*pow(sin(omega*t),2) - 6*pow(x[0],2)*pow(x[1],2)*pow((x[1] - 1),2)*omega*cos(omega*t) + 2*pow(x[0],2)*x[1]*sin(omega*t) - pow(x[0],2)*sin(omega*t) + 2*x[0]*pow(x[1],2)*pow((x[1] - 1),2)*omega*cos(omega*t) - 2*x[0]*x[1]*sin(omega*t) + x[0]*sin(omega*t)"), t=0, nu=self.nu, omega = self.omega )

		self.v = Expression((
			"sin(omega*t)*x[0]*x[0]*(1 - x[0])*(1 - x[0])*2*x[1]*(1 - x[1])*(2*x[1] - 1)", 
			"sin(omega*t)*x[1]*x[1]*(1 - x[1])*(1 - x[1])*2*x[0]*(1 - x[0])*(1 - 2*x[0])"), omega = self.omega, t = 0)
		self.vdot = Expression((
			"omega*cos(omega*t)*x[0]*x[0]*(1 - x[0])*(1 - x[0])*2*x[1]*(1 - x[1])*(2*x[1] - 1)",
			"omega*cos(omega*t)*x[1]*x[1]*(1 - x[1])*(1 - x[1])*2*x[0]*(1 - x[0])*(1 - 2*x[0])"), omega = self.omega, t = 0)
		self.p =  Expression(("sin(omega*t)*x[0]*(1-x[0])*x[1]*(1-x[1])"), omega = self.omega, t = 0)

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
