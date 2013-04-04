import unittest
import numpy as np
import scipy.sparse as sps

class TestSmaMinTexFunctions(unittest.TestCase):

	def setUp(self):
		# for NSE solution
		self.Nlist = [8, 16, 32]#, 64] # set up for three values that doubles
		self.tcur = 0.5
		self.OC = 3 - 0.01 # = order of convergence minus a threshold

	def test_collectB2(self):

		from smamin_utils import col_columns_atend
		
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

	def test_rearrange_matrices(self):
		"""check the algorithm rearranging for B2, solve the

		rearranged system and sort back the solution vector 
		The rearr. is done by swapping columns in the coeff matrix,
		thus the resort is done by swapping the entries of the solution
		vector"""

		from dolfin import *
		import test_time_schemes as tts
		import dolfin_to_nparrays as dtn
		from smamin_utils import col_columns_atend, revert_sort_tob2
		import smartminex_tayhoomesh

		N = 32 
		mesh = smartminex_tayhoomesh.getmake_mesh(N)

		V = VectorFunctionSpace(mesh, "CG", 2)
		Q = FunctionSpace(mesh, "CG", 1)

		velbcs = tts.setget_velbcs_zerosq(mesh, V)

		bcinds = []
		for bc in velbcs:
			bcdict = bc.get_boundary_values()
			bcinds.extend(bcdict.keys())

		# indices of the inner velocity nodes
		invinds = np.setdiff1d(range(V.dim()),bcinds)

		Ma, Aa, BTa, Ba, MPa = dtn.get_sysNSmats(V, Q)

		Mc = Ma[invinds,:][:,invinds]
		Ac = Aa[invinds,:][:,invinds]
		Bc  = Ba[:,invinds]
		BTc = BTa[invinds,:]

		B2BubInds = smartminex_tayhoomesh.get_B2_bubbleinds(N, V, mesh)
		#B2BubInds = np.array([2,4])
		# we need the B2Bub indices in the reduced setting vc
		# this gives a masked array of boolean type
		B2BubBool = np.in1d(np.arange(V.dim())[invinds], B2BubInds)
		#B2BubInds = np.arange(len(B2BubIndsBool

		Nv = len(invinds)
		Np = Q.dim()
		dt = 1.0/N

		# the complement of the bubble index in the inner nodes
		BubIndC = ~B2BubBool #np.setdiff1d(np.arange(Nv),B2BubBool)
		# the bubbles as indices
		B2BI = np.arange(len(B2BubBool))[B2BubBool]

		# Reorder the matrices for smart min ext
		MSme = col_columns_atend(Mc, B2BI)
		BSme = col_columns_atend(Bc, B2BI)

		B1Sme = BSme[:,:Nv-(Np-1)]
		B2Sme = BSme[:,Nv-(Np-1):]

		M1Sme = MSme[:,:Nv-(Np-1)]
		M2Sme = MSme[:,Nv-(Np-1):]

		IterA1 = sps.hstack([sps.hstack([M1Sme,dt*M2Sme]), -dt*BTc[:,1:]])

		IterA2 = sps.hstack([sps.hstack([B1Sme[1:,:],dt*B2Sme[1:,:]]),
			sps.csr_matrix((Np-1, (Np-1)))])
		# The rearranged coefficient matrix
		IterARa = sps.vstack([IterA1,IterA2])

		IterAqq = sps.hstack([Mc,-dt*BTc[:,1:]])
		IterAp = sps.hstack([Bc[1:,:],sps.csr_matrix((Np-1,Np-1))])
		# The actual coefficient matrix
		IterA  = sps.vstack([IterAqq,IterAp])

		rhs = np.random.random((Nv+Np-1,1))

		SolActu = sps.linalg.spsolve(IterA,rhs)
		SolRa   = sps.linalg.spsolve(IterARa,rhs)

		# Sort it back 
		# manually 
		SortBack = np.zeros((Nv+Np-1,1))
		# q1
		SortBack[BubIndC,0] = SolRa[:Nv-(Np-1)]
		# tq2
		SortBack[B2BI,0] = dt*SolRa[Nv-(Np-1):Nv]
		SortBack[Nv:,0] = SolRa[Nv:]

		SolRa = np.atleast_2d(SolRa).T

		# and by function
		SolRa[Nv-(Np-1):Nv,0] = dt*SolRa[Nv-(Np-1):Nv,0]
		SB2 = revert_sort_tob2(SolRa[:Nv,],B2BI)
		SB2 = np.vstack([SB2,SolRa[Nv:,]])
		# SB2v = np.atleast_2d(SB2)

		SolActu = np.atleast_2d(SolActu)
		SortBack = np.atleast_2d(SortBack).T

		#print SolActu
		#print SolRa 
		#print SortBack 
		#print SB2

		self.assertTrue(np.allclose(SolActu,SortBack,atol=1e-6))
		self.assertTrue(np.allclose(SolActu,SB2.T,atol=1e-6))


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

		eoca = np.zeros(len(self.Nlist))
		eocc = np.zeros(len(self.Nlist))

		for k, N in enumerate(self.Nlist):
			PrP = tts.ProbParams(N,12)
			# get system matrices as np.arrays
			Ma, Aa, BTa, Ba, MPa= dtn.get_sysNSmats(PrP.V, PrP.Q)

			tcur = self.tcur

			PrP.fv.t  = tcur
			PrP.fv.nu = nu   # nu = 0, for Eulerian Flow - no diffusion

			fv, fp = dtn.setget_rhs(PrP.V, PrP.Q, PrP.fv, PrP.fp)

			Mc, Ac, BTc, Bc, fvbc, fpbc, bcinds, bcvals, invinds = \
					dtn.condense_sysmatsbybcs(Ma,Aa,BTa,Ba,fv,fp,PrP.velbcs)

			v, vdot, p = PrP.v, PrP.vdot, PrP.p


			vc = get_funcexpr_as_vec(v, PrP.V, t=tcur)
			pc = get_funcexpr_as_vec(p, PrP.Q, t=tcur)

			vdotc = get_funcexpr_as_vec(vdot, PrP.V, t=tcur )

			v.t = tcur
			vf = project(v, PrP.V)
			ConV  = dtn.get_convvec(vf, PrP.V)

			resV = Ma*vdotc + nu*Aa*vc + ConV[:,0] - BTa*pc - fv[:,0]
			resVc = Mc*vdotc[invinds] + nu*Ac*vc[invinds] + ConV[invinds,0] - BTc*pc - fvbc[:,0] - fv[invinds,0]

			eoca[k] = (np.sqrt(np.dot(np.atleast_2d(resV[invinds]),Mc*np.atleast_2d(resV[invinds]).T)))
			eocc[k] = (np.sqrt(np.dot(np.atleast_2d(resVc),Mc*np.atleast_2d(resVc).T)))

		eocs = np.array([log(eoca[0]/eoca[1]), log(eoca[1]/eoca[2]), log(eocc[0]/eocc[1]), log(eocc[1]/eocc[2])])

		OC = self.OC

		print eocs
		print np.array([OC*log(2)]*4)

		self.assertTrue((eocs > [np.array([OC*log(2)])]*4 ).all())

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

		eoca = np.zeros(len(self.Nlist))
		eocc = np.zeros(len(self.Nlist))

		for k, N in enumerate(self.Nlist):
			PrP = tts.ProbParams(N,8)
			# get system matrices as np.arrays
			Ma, Aa, BTa, Ba, MPa = dtn.get_sysNSmats(PrP.V, PrP.Q)

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

			eoca[k] = (np.sqrt(np.dot(np.atleast_2d(resV[invinds]),Mc*np.atleast_2d(resV[invinds]).T)))
			eocc[k] = (np.sqrt(np.dot(np.atleast_2d(resVc),Mc*np.atleast_2d(resVc).T)))

		eocs = np.array([log(eoca[0]/eoca[1]), log(eoca[1]/eoca[2]), log(eocc[0]/eocc[1]), log(eocc[1]/eocc[2])])

		OC = self.OC

		print eocs
		print np.array([OC*log(2)]*4)

		self.assertTrue((eocs > [np.array([OC*log(2)])]*4 ).all())


if __name__ == '__main__':
    unittest.main()

