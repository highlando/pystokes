from dolfin import *
import numpy as np
import scipy.sparse as sps
from scipy.linalg import qr


def qr_impeuler(Mc,Ac,BTc,Bc,fvc,fp,vp_init,PrP,TsP=None):
	""" with BTc[:-1,:] = Q*[R ; 0] 
	we define ~M = Q*M*Q.T , ~A = Q*A*Q.T , ~V = Q*v
	and condense the system accordingly """

	Nts, t0, tE = TsP.Nts, TsP.t0, TsP.tE

	dt = (t0-tE)/Nts

	Nv = len(fvc)
	Np = len(fp)

	v, p   = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None)

	tcur = t0

	TsP.UpFiles.u_file << v, tcur
	TsP.UpFiles.p_file << p, tcur

	Qm, Rm = qr(BTc.todense()[:,:-1],mode='full')

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

	for etap in range(1,11):
		for i in range(Nts/10):
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
		TsP.UpFiles.u_file << v, tcur
		TsP.UpFiles.p_file << p, tcur
		print '%d of %d time steps completed \n' % (etap*Nts/10,Nts) 
		
	return

def plain_impeuler(Mc,Ac,BTc,Bc,fvc,fp,vp_init,PrP,TsP):

	Nts, t0, tE = TsP.Nts, TsP.t0, TsP.tE

	dt = (t0-tE)/Nts

	Nv = len(fvc)
	Np = len(fp)

	v, p   = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None)

	tcur = t0

	TsP.UpFiles.u_file << v, tcur
	TsP.UpFiles.p_file << p, tcur

	IterAv = sps.hstack([Mc-dt*Ac,dt*BTc])
	IterAp = sps.hstack([dt*Bc,sps.csr_matrix((Np,Np))])
	IterA  = sps.vstack([IterAv,IterAp]).todense()[:-1,:-1]

	vp_old = vp_init
	for etap in range(1,11):
		for i in range(Nts/10):
			tcur = tcur + dt
			#iterateeee
			Iterrhs = np.vstack([Mc*vp_old[:Nv,],np.zeros((Np-1,1))]) \
					+ dt*np.vstack([fvc,fp[:-1,]])
			vp_new = np.linalg.solve(IterA,Iterrhs)
			vp_old = vp_new


		print '%d of %d time steps completed \n' % (etap*Nts/10,Nts) 
		v, p = expand_vp_dolfunc(PrP, vp=vp_new, vc=None, pc=None)

		TsP.UpFiles.u_file << v, tcur
		TsP.UpFiles.p_file << p, tcur
		
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
