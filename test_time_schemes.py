from dolfin import *
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import os, glob

import dolfin_to_nparrays as dtn
reload(dtn)
import time_int_schemes as tis
reload(tis)
		
import smartminex_tayhoomesh 
reload(smartminex_tayhoomesh)
class TimestepParams(object):
	def __init__(self, method, N):
		self.t0 = 0
		self.tE = 3.0
		self.Ntslist = [256, 512]#, 1024]#, 2048]
		self.SampInt = self.Ntslist[0]/16
		self.method = method
		self.UpFiles = UpFiles(method)
		self.Residuals = NseResiduals()
		self.linatol = 0#1e-5 #1e-8   # 0 for direct sparse solver
		self.ParaviewOutput = True
		#self.PickleFile = 'pickles/NTs%dto%dMesh%d' % (self.Ntslist[0], self.Ntslist[-1], N) + method

def solve_stokesTimeDep(method=None, N=None, NtsList=None, LinaTol=None):
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
			3:'HalfExpEulSmaMin',
			4:'HalfExpEulSmaMinSplit'}

	# instantiate object containing mesh, V, Q, velbcs, invinds
	PrP = ProbParams(N)
	# instantiate the Time Int Parameters
	TsP = TimestepParams(methdict[method], N)
	if NtsList is not None:
		TsP.Ntslist = NtsList
	if LinaTol is not None:
		TsP.linatol = LinaTol

	print 'Mesh parameter N = %d' % N
	print 'You have chosen %s for time integration' % methdict[method]
	print 'The tolerance for the linear solver is %e' %TsP.linatol

	#prepare for pickling the residuals
	# fpic = file(TsP.PickleFile,'w')
	# pickle.dump(TsP.Residuals, fpic)

	# get system matrices as np.arrays
	Ma, Aa, BTa, Ba = dtn.get_sysNSmats(PrP.V, PrP.Q)
	fv, fp = dtn.setget_rhs(PrP.V, PrP.Q, PrP.fv, PrP.fp)

	Mc, Ac, BTc, Bc, fvbc, fpbc, bcinds, bcvals, invinds = \
			dtn.condense_sysmatsbybcs(Ma,Aa,BTa,Ba,fv,fp,PrP.velbcs)
	
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
		else:
			# get the indices of the B2-part
			B2Inds = smartminex_tayhoomesh.get_B2_bubbleinds(N, PrP.V, PrP.mesh)
			# the B2 inds wrt to inner nodes
			# this gives a masked array of boolean type
			B2BoolInv = np.in1d(np.arange(PrP.V.dim())[PrP.invinds], B2BubInds)
			# this as indices
			B2BI = np.arange(len(B2BoolInv))[B2BoolInv]
			# Reorder the matrices for smart min ext
			MSme = col_columns_atend(Mc, B2BI)
			BSme = col_columns_atend(Bc, B2BI)

			if method == 3:
				tis.halfexp_euler_smarminex(Mc,Ac,BTc,Bc,fvbc,fpbc,
						vp_init,B2BoolInv,PrP,TsP=TsP)
			elif method == 4:
				tis.halfexp_euler_smarminex_wminresprec(Mc,Ac,BTc,Bc,fvbc,fpbc,
						vp_init,B2BoolInv,PrP,TsP=TsP)
			elif method == 5:  #no removal of the pressure freedom
				tis.halfexp_euler_smarminex_fpsplit(Mc,Ac,BTc,Bc,fvbc,fpbc,
						vp_init,B2BoolInv,PrP,TsP=TsP)

		# Output only in first iteration!
		TsP.ParaviewOutput = False
	
	plot_errs_res(TsP)
	save_simu(TsP, PrP)
		
	#vp_stat = np.linalg.solve(sadSysmat[:-1,:-1],np.vstack([fvc,fp[:-1],]))
	#v, p = expand_vp_dolfunc(invinds,velbcs,V,Q,
	#		vp=vp_stat,vc=None,pc=None)
	# u_file << v, 1
	#p_file << p, 1

	return 

def save_simu(TsP, PrP):
	import json

	DictOfVals = {'SpaceDiscParam': PrP.N,
			'Omega': PrP.omega,
			'TimeInterval':[TsP.t0,TsP.tE],
			'TimeDiscs': TsP.Ntslist,
			'LinaTol': TsP.linatol,
			'TimeIntMeth': TsP.method,
			'ContiRes': TsP.Residuals.ContiRes,
			'VelEr': TsP.Residuals.VelEr,
			'PEr': TsP.Residuals.PEr}

	JsFile = 'json/Omeg%dTol%0.0eNTs%dto%dMesh%d' % (DictOfVals['Omega'], TsP.linatol, TsP.Ntslist[0], TsP.Ntslist[-1], PrP.N) +TsP.method + '.json'

	f = open(JsFile, 'w')
	f.write(json.dumps(DictOfVals))

	print 'Simulation data stored in "' + JsFile + '"'

	return

def load_json_dicts(StrToJs):
	import json
	fjs = open(StrToJs)
	JsDict = json.load(fjs)
	return JsDict


def merge_json_dicts(CurDi,DiToAppend):
	import json

	Jsc = load_json_dicts(CurDi)
	Jsa = load_json_dicts(DiToAppend)

	if Jsc['SpaceDiscParam'] != Jsa['SpaceDiscParam'] or Jsc['Omega'] != Jsa['Omega']:
		raise Warning('Space discretization or omega do not match')

	Jsc['TimeDiscs'].extend(Jsa['TimeDiscs'])
	Jsc['ContiRes'].extend(Jsa['ContiRes'])
	Jsc['VelEr'].extend(Jsa['VelEr'])
	Jsc['PEr'].extend(Jsa['PEr'])

	JsFile = 'json/MrgdOmeg%dTol%0.0eNTs%dto%dMesh%d' % (Jsc['LinaTol'], Jsc['TimeDiscs'][0], Jsc['TimeDiscs'][-1], Jsc['SpaceDiscParam']) +Jsc['TimeIntMeth'] + '.json'

	f = open(JsFile, 'w')
	f.write(json.dumps(Jsc))

	print '"Merged data stored in ' + JsFile + '"'

	return Jsc


def jsd_plot_errs(JsDict):

	JsDict = load_json_dicts(JsDict)

	plt.close('all')
	for i in range(len(JsDict['TimeDiscs'])):
		leg = 'NTs = $%d$' % JsDict['TimeDiscs'][i]
		plt.figure(1)
		plt.plot(JsDict['ContiRes'][i],label=leg)
		plt.title(JsDict['TimeIntMeth']+': continuity eqn residual')
		plt.legend()
		plt.figure(2)
		plt.plot(JsDict['VelEr'][i],label=leg)
		plt.title(JsDict['TimeIntMeth']+': Velocity error')
		plt.legend()
		plt.figure(3)
		plt.plot(JsDict['PEr'][i],label=leg)
		plt.title(JsDict['TimeIntMeth']+': Pressure error')
		plt.legend()

	plt.show()

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

		self.N = N
		self.V = VectorFunctionSpace(self.mesh, "CG", 2)
		self.Q = FunctionSpace(self.mesh, "CG", 1)
		self.velbcs = setget_velbcs_zerosq(self.mesh, self.V)
		self.Pdof = 0  #dof removed in the p approximation
		self.omega = 3
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
