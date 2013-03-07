import unittest
import numpy as np
import scipy.sparse as sps

class TestSequenceFunctions(unittest.TestCase):

	def test_collectB2(self):

		from time_int_schemes import col_columns_atend
		
		N = 100
		n = 7
		col = np.arange(0,N,n)
		v = np.arange(N)
		mat = sps.spdiags(v,[0],N,N+len(col))

		MatRa = col_columns_atend(mat,col)
		vra = np.append(v,col)
		va  = np.append(v,0*col)
		vra[col] = 0

		self.assertTrue(np.allclose(MatRa*vra,mat*va))

	def test_nse_solution(self):
		from dolfin import *
		import test_time_schemes as tts

		PrP = ProbParams(N)
		# get system matrices as np.arrays
		Ma, Aa, BTa, Ba = dtn.get_sysNSmats(PrP.V, PrP.Q)

		PrP.fv.t  = tcur
		PrP.fv.nu = nu   # nu = 0, for Eulerian Flow - no diffusion

		fv, fp = dtn.setget_rhs(PrP.V, PrP.Q, PrP.fv, PrP.fp)

		Mc, Ac, BTc, Bc, fvbc, fpbc, bcinds, bcvals, invinds = \
				dtn.condense_sysmatsbybcs(Ma,Aa,BTa,Ba,fv,fp,PrP.velbcs)

		v, p = PrP.v, PrP.p

		def get_funcexpr_as_condensedvec(u,U,invinds,t=None):
			if t is not None:
				u.t = t
			ua = project(u,U)
			ua = ua.vector().array()
			uc = np.atleast_2d(ua[invinds].T)
			return uc

		vcc = get_funcexpr_as_condensedvec(vCur, PrP.V, invinds, t=tcur)
		pcc = get_funcexpr_as_condensedvec(pCur, PrP.Q,
				range(PrP.Q.dim()-1), t=tcur)
		vnc = get_funcexpr_as_condensedvec(vNex, PrP.V, invinds, t=tcur+dt)
		ConV  = dtn.get_convvec(v, PrP.V)

		resVc = Mc*(vnc - vcc) + dt*(nu*Ac*vcc 


if __name__ == '__main__':
    unittest.main()

