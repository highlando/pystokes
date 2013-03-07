import unittest
import numpy as np
import scipy.sparse as sps

class TestSmaMinTexFunctions(unittest.TestCase):

	def setUp(self):
		# for NSE solution
		self.Nlist = [8, 16, 32]
		self.tcur = 0.5
		self.OC = 6 # = order of convergence + 2, for not scaling by M^-1 

	def test_collectB2(self):

		from time_int_schemes import col_columns_atend
		
		N = 100
		n = 7

		col = np.arange(0,N,n).astype(int)

		v = np.arange(N)
		v = np.append(v,0*col)

		colI = np.in1d(range(len(v)),col)

		mat = sps.spdiags(v,[0],N,N+len(col))
		MatRa = col_columns_atend(mat, col)

		vra = np.r_[v[~colI],v[col]]

		self.assertTrue(np.allclose(MatRa*vra, mat*v))

	def test_nse_solution(self):
		from dolfin import project
		import test_time_schemes as tts
		import dolfin_to_nparrays as dtn
		from numpy import log

		nu   = 1

		def get_funcexpr_as_vec(u, U, invinds=None, t=None):
			if t is not None:
				u.t = t
			if invinds is None:
				invinds = range(U.dim())
			ua = project(u,U)
			ua = ua.vector()
			ua = ua.array()
			#uc = np.atleast_2d(ua[invinds].T)
			return ua

		eoca = []
		eocc = []

		for k, N in enumerate(self.Nlist):
			PrP = tts.ProbParams(N)
			# get system matrices as np.arrays
			Ma, Aa, BTa, Ba = dtn.get_sysNSmats(PrP.V, PrP.Q)

			tcur = self.tcur

			PrP.fv.t  = tcur
			PrP.fv.nu = nu   # nu = 0, for Eulerian Flow - no diffusion

			fv, fp = dtn.setget_rhs(PrP.V, PrP.Q, PrP.fv, PrP.fp)

			Mc, Ac, BTc, Bc, fvbc, fpbc, bcinds, bcvals, invinds = \
					dtn.condense_sysmatsbybcs(Ma,Aa,BTa,Ba,fv,fp,PrP.velbcs)

			v, vdot, p = PrP.v, PrP.vdot, PrP.p


			vc = get_funcexpr_as_vec(v, PrP.V, t=tcur)
			pc = get_funcexpr_as_vec(p, PrP.Q, t=tcur)

			vdotc = get_funcexpr_as_vec(vdot, PrP.V, t=tcur)

			v.t = tcur
			vf = project(v, PrP.V)
			ConV  = dtn.get_convvec(vf, PrP.V)

			resV = Ma*vdotc + nu*Aa*vc + ConV[:,0] - BTa*pc - fv[:,0]
			resVc = Mc*vdotc[invinds] + nu*Ac*vc[invinds] + ConV[invinds,0] - BTc*pc - fvbc[:,0] - fv[invinds,0]

			eoca.append(np.dot(np.atleast_2d(resV[invinds]),Mc*np.atleast_2d(resV[invinds]).T))
			eocc.append(np.dot(np.atleast_2d(resVc),Mc*np.atleast_2d(resVc).T))

		eocs = [log(eoca[0]/eoca[1]), log(eoca[1]/eoca[2]), log(eocc[0]/eocc[1]), log(eocc[1]/eocc[2])] 
		print eocs

		OC = self.OC

		self.assertTrue(eocs > [OC*log(2)]*4 )

	def test_eul_solution(self):
		from dolfin import project
		import test_time_schemes as tts
		import dolfin_to_nparrays as dtn
		from numpy import log

		nu   = 0

		def get_funcexpr_as_vec(u, U, invinds=None, t=None):
			if t is not None:
				u.t = t
			if invinds is None:
				invinds = range(U.dim())
			ua = project(u,U)
			ua = ua.vector()
			ua = ua.array()
			#uc = np.atleast_2d(ua[invinds].T)
			return ua

		eoca = []
		eocc = []

		for k, N in enumerate(self.Nlist):
			PrP = tts.ProbParams(N)
			# get system matrices as np.arrays
			Ma, Aa, BTa, Ba = dtn.get_sysNSmats(PrP.V, PrP.Q)

			tcur = self.tcur

			PrP.fv.t  = tcur
			PrP.fv.nu = nu   # nu = 0, for Eulerian Flow - no diffusion

			fv, fp = dtn.setget_rhs(PrP.V, PrP.Q, PrP.fv, PrP.fp)

			Mc, Ac, BTc, Bc, fvbc, fpbc, bcinds, bcvals, invinds = \
					dtn.condense_sysmatsbybcs(Ma,Aa,BTa,Ba,fv,fp,PrP.velbcs)

			v, vdot, p = PrP.v, PrP.vdot, PrP.p


			vc = get_funcexpr_as_vec(v, PrP.V, t=tcur)
			pc = get_funcexpr_as_vec(p, PrP.Q, t=tcur)

			vdotc = get_funcexpr_as_vec(vdot, PrP.V, t=tcur)

			v.t = tcur
			vf = project(v, PrP.V)
			ConV  = dtn.get_convvec(vf, PrP.V)

			resV = Ma*vdotc + nu*Aa*vc + ConV[:,0] - BTa*pc - fv[:,0]
			resVc = Mc*vdotc[invinds] + nu*Ac*vc[invinds] + ConV[invinds,0] - BTc*pc - fvbc[:,0] - fv[invinds,0]

			eoca.append(np.dot(np.atleast_2d(resV[invinds]),Mc*np.atleast_2d(resV[invinds]).T))
			eocc.append(np.dot(np.atleast_2d(resVc),Mc*np.atleast_2d(resVc).T))

		eocs = [log(eoca[0]/eoca[1]), log(eoca[1]/eoca[2]), log(eocc[0]/eocc[1]), log(eocc[1]/eocc[2])] 
		
		OC = self.OC

		self.assertTrue(eocs > [OC*log(2)]*4 )


if __name__ == '__main__':
    unittest.main()

