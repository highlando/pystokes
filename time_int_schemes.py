from dolfin import *
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import krypy.linsys
from scipy.linalg import qr

import dolfin_to_nparrays as dtn

###
# solve M\dot v + K(v) -B'p = fv
#                 Bv      = fpbc
###


def halfexp_euler_smarminex(MSme,BSme,MP,FvbcSme,FpbcSme,B2BoolInv,PrP,TsP,vp_init=None,qqpq_init=None):
	"""halfexplicit euler for the NSE in index 1 formulation
	
	"""

	N, Pdof = PrP.N, PrP.Pdof
	Nts, t0, tE, dt, Nv, Np = init_time_stepping(PrP,TsP)
	tcur = t0

	Npc = Np-1

	# remove the p - freedom
	if Pdof == 0:
		BSme  = BSme[1:,:]
		FpbcSmeC = FpbcSme[1:,]
		MPc = MP[1:,:][:,1:]
	else:
		BSme  = sps.vstack([BSme[:Pdof,:],BSme[Pdof+1:,:]])
		raise Warning('TODO: Implement this')

	B1Sme = BSme[:,:Nv-(Np-1)]
	B2Sme = BSme[:,Nv-(Np-1):]

	M1Sme = MSme[:,:Nv-(Np-1)]
	M2Sme = MSme[:,Nv-(Np-1):]
	
	#### The matrix to be solved in every time step
	#
	# 		M11    M12  B2'  0          q1
	# 		M21    M22  B1'  0         dt*tq2
	# 		B1     B2   0    0   *    -dt*p     = rhs
	# 		B1     0    0    B2 	    q2  
	# 						    		 
	# cf. preprint
	#

	IterA1 = sps.hstack([MSme,BSme.T])
	IterA2 = sps.hstack([BSme, sps.csr_matrix((Np-1, Np-1))])
	IterASp = sps.vstack([IterA1,IterA2])

	IterA3 = sps.hstack([sps.hstack([B1Sme,sps.csr_matrix((Np-1,2*(Np-1)))]),
		B2Sme])

	IterA = sps.vstack([
		sps.hstack([IterASp, sps.csr_matrix((Nv+Np-1,Np-1))]),
				IterA3])

	## Preconditioning ...
	#
	if TsP.SadPtPrec:
		MLump = np.atleast_2d(MSme.diagonal()).T
		MLump2 =  MLump[-(Np-1):,]
		MLumpI = 1./MLump
		MLumpI1 = MLumpI[:-(Np-1),]
		MLumpI2 = MLumpI[-(Np-1):,]
		def PrecByB2(qqpq):
			qq = MLumpI*qqpq[:Nv,] 

			p  = qqpq[Nv:-(Np-1),]
			p  = spsla.spsolve(B2Sme, p)
			p  = MLumpI2*np.atleast_2d(p).T
			p  = spsla.spsolve(B2Sme.T,p)
			p  = np.atleast_2d(p).T

			q2 = qqpq[-(Np-1):,]
			q2 = spsla.spsolve(B2Sme,q2)
			q2 = np.atleast_2d(q2).T

			return np.vstack([np.vstack([qq, -p]), q2])
		
		MGmr = spsla.LinearOperator( (Nv+2*(Np-1),Nv+2*(Np-1)), matvec=PrecByB2, dtype=np.float32 )
		TsP.Ml = MGmr
	
	def smamin_ip(qqpq1, qqpq2):
		"""inner product that 'inverts' the preconditioning
		
		and weights the conti residuals 
		for better comparability of the residuals, i.e. the tolerances
		"""
		def _inv_prec(qqpq):
			qq = MLump*qqpq[:Nv,] 
			p  = qqpq[Nv:-(Np-1),]
			p  = B2Sme.T*p
			p  = MLump2*p
			p  = B2Sme*p
			q2 = qqpq[-(Np-1):,]
			q2 = B2Sme*q2
			return qq, p, q2

		if TsP.SadPtPrec:
			qq1,p1,q21 = _inv_prec(qqpq1)
			qq2,p2,q22 = _inv_prec(qqpq2)
		else:
			qq1,p1,q21 = qqpq1[:Nv,], qqpq1[Nv:-(Np-1),], qqpq1[-(Np-1):,]
			qq2,p2,q22 = qqpq2[:Nv,], qqpq2[Nv:-(Np-1),], qqpq2[-(Np-1):,]

		return np.dot(qq1.T.conj(),qq2) + 0.5*np.dot(p1.T.conj(),p2) + 0.5*np.dot(q21.T.conj(),q22)

	v, p   = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None, pdof=None)
	TsP.UpFiles.u_file << v, tcur
	TsP.UpFiles.p_file << p, tcur

	if qqpq_init is None:
		vp_old = np.copy(vp_init)
		q1_old = vp_init[~B2BoolInv,]
		q2_old = vp_init[B2BoolInv,]
		# initial value for tq2
		ConV, CurFv = get_conv_curfv_rearr(v,PrP,tcur,B2BoolInv)
		tq2_old = spsla.spsolve(M2Sme[-(Np-1):,:], CurFv[-(Np-1):,])
		#tq2_old = MLumpI2*CurFv[-(Np-1):,]
		tq2_old = np.atleast_2d(tq2_old).T
		# state vector of the smaminex system : [ q1^+, tq2^c, p^c, q2^+] 
		qqpq_old = np.zeros((Nv+2*(Np-1),1))
		qqpq_old[:Nv-(Np-1),] = q1_old
		qqpq_old[Nv-(Np-1):Nv,] = dt*tq2_old 
		qqpq_old[Nv:Nv+Np-1,] = -dt*vp_old[Nv:,]
		qqpq_old[Nv+Np-1:,] = q2_old
	else:
		qqpq_old = qqpq_init
		q1_old = qqpq_old[:Nv-Npc,]

	ContiRes, VelEr, PEr = [], [], []

	for etap in range(1,TsP.NOutPutPts +1 ):
		for i in range(Nts/TsP.NOutPutPts):
			ConV, CurFv = get_conv_curfv_rearr(v,PrP,tcur,B2BoolInv)

			gdot = np.zeros((Np-1,1)) # TODO: implement \dot g

			Iterrhs = np.vstack([M1Sme*q1_old, B1Sme*q1_old]) \
						+ dt*np.vstack([FvbcSme+CurFv-ConV,gdot])
			Iterrhs = np.vstack([Iterrhs,FpbcSmeC])

			if TsP.linatol == 0:
				q1_tq2_p_q2_new = spsla.spsolve(IterA,Iterrhs) 
				qqpq_old = np.atleast_2d(q1_tq2_p_q2_new).T

			else:
				q1_tq2_p_q2_new = krypy.linsys.gmres(IterA, Iterrhs,
						x0=qqpq_old, Ml=TsP.Ml, Mr=TsP.Mr, 
						inner_product=smamin_ip,
						tol=TsP.linatol, maxiter=TsP.MaxIter)
				qqpq_old = np.atleast_2d(q1_tq2_p_q2_new['xk'])
				if q1_tq2_p_q2_new['info'] != 0:
					print q1_tq2_p_q2_new['relresvec'][-5:]
					raise Warning('no convergence')
				
				print 'Needed %d of max  %r iterations: final relres = %e' %(len(q1_tq2_p_q2_new['relresvec']), TsP.MaxIter, q1_tq2_p_q2_new['relresvec'][-1] )

			q1_old = qqpq_old[:Nv-(Np-1),]

			# Extract the 'actual' velocity and pressure
			vc = np.zeros((Nv,1))
			vc[~B2BoolInv,] = qqpq_old[:Nv-(Np-1),]
			vc[B2BoolInv,] = qqpq_old[-(Np-1):,]
			print np.linalg.norm(vc)
			pc = -1.0/dt*qqpq_old[Nv:Nv+Np-1,]

			v, p = expand_vp_dolfunc(PrP, vp=None, vc=vc, pc=pc, pdof = Pdof)
			
			tcur += dt

			# the errors and residuals 
			vCur, pCur = PrP.v, PrP.p 
			vCur.t = tcur
			pCur.t = tcur - dt

			ContiRes.append(comp_cont_error(v,FpbcSme,PrP.Q))
			VelEr.append(errornorm(vCur,v))
			PEr.append(errornorm(pCur,p))

			if i + etap == 1 and TsP.SaveIniVal:
				from scipy.io import savemat
				dname = 'IniValSmaMinN%s' % N
				savemat(dname, { 'qqpq_old': qqpq_old })



		print '%d of %d time steps completed ' % (etap*Nts/TsP.NOutPutPts, Nts) 

		if TsP.ParaviewOutput:
			TsP.UpFiles.u_file << v, tcur
			TsP.UpFiles.p_file << p, tcur

	return

def halfexp_euler_ind2ra(MSme,BSme,MP,FvbcSme,FpbcSme,vp_init,B2BoolInv,PrP,TsP):
	"""halfexplicit euler for the NSE in index 2 formulation

	but with the smamin rearrangement for preconditioning...
	"""

	N, Pdof = PrP.N, PrP.Pdof

	Nts, t0, tE, dt, Nv, Np = init_time_stepping(PrP,TsP)
	tcur = t0

	Npc = Np-1

	# remove the p - freedom
	if Pdof == 0:
		BSme  = BSme[1:,:]
		FpbcSmeC = FpbcSme[1:,]
		MPc = MP[1:,:][:,1:]
	else:
		BSme  = sps.vstack([BSme[:Pdof,:],BSme[Pdof+1:,:]])
		raise Warning('TODO: Implement this')

	B1Sme = BSme[:,:Nv-(Np-1)]
	B2Sme = BSme[:,Nv-(Np-1):]

	M1Sme = MSme[:,:Nv-(Np-1)]
	M2Sme = MSme[:,Nv-(Np-1):]
	
	#### The matrix to be solved in every time step
	#
	# 		M11    M12  B2'          	q1
	# 		M21    M22  B1'          	q2
	# 		B1     B2   0       *    -dt*p     = rhs
	# 						    		 
	# cf. preprint
	#

	IterA1 = sps.hstack([MSme,BSme.T])

	# Multiply by -dt for symmetry
	IterA2 = sps.hstack([BSme, sps.csr_matrix((Np-1, Np-1))])

	IterASp = sps.vstack([IterA1,IterA2])
	
	## Preconditioning ...
	#
	MLump = np.atleast_2d(MSme.diagonal()).T
	MLump2 =  MLump[-(Np-1):,]
	MLumpI = 1./MLump
	MLumpI1 = MLumpI[:-(Np-1),]
	MLumpI2 = MLumpI[-(Np-1):,]
	def PrecByB2(qqp):
		qq = MLumpI*qqp[:Nv,] 

		p  = qqp[Nv:,]
		p  = spsla.spsolve(B2Sme, p)
		p  = MLumpI2*np.atleast_2d(p).T
		p  = spsla.spsolve(B2Sme.T,p)
		p  = np.atleast_2d(p).T

		return np.vstack([qq, -p])
	
	MGmr = spsla.LinearOperator( (Nv+(Np-1),Nv+(Np-1)), matvec=PrecByB2, dtype=np.float32 )
	TsP.Ml = MGmr
	
	def ind2ra_ip(qqp1, qqp2):
		"""inner product that 'inverts' the preconditioning
		
		and doubles the weight on the conti eqn
		for better comparability of the residuals, i.e. the tolerances
		"""
		def _inv_prec(qqp):
			qq = MLump*qqp[:Nv,] 
			p  = qqp[Nv:,]
			p  = B2Sme.T*p
			p  = MLump2*p
			p  = B2Sme*p
			return qq, p

		qq1,p1 = _inv_prec(qqp1)
		qq2,p2 = _inv_prec(qqp2)

		return np.dot(qq1.T.conj(),qq2) + np.dot(p1.T.conj(),p2) 
	

	v, p   = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None, pdof=None)
	TsP.UpFiles.u_file << v, tcur
	TsP.UpFiles.p_file << p, tcur

	vp_old = np.copy(vp_init)
	q1_old = vp_init[~B2BoolInv,]
	q2_old = vp_init[B2BoolInv,]

	ConV, CurFv = get_conv_curfv_rearr(v,PrP,tcur,B2BoolInv)

	# state vector of the system : [ q1^+, q2^+, p^c] 
	qqp_old = np.zeros((Nv+(Np-1),1))
	qqp_old[:Nv-(Np-1),] = q1_old
	qqp_old[Nv-(Np-1):Nv,] = q2_old
	qqp_old[Nv:Nv+Np-1,] = vp_old[Nv:,]

	ContiRes, VelEr, PEr = [], [], []

	for etap in range(1,TsP.NOutPutPts +1 ):
		for i in range(Nts/TsP.NOutPutPts):
			ConV, CurFv = get_conv_curfv_rearr(v,PrP,tcur,B2BoolInv)

			Iterrhs = np.vstack([M1Sme*q1_old + M2Sme*q2_old 
				+ dt*(FvbcSme+CurFv-ConV),
					FpbcSmeC])

			if TsP.linatol == 0:
				q1_q2_p_new = spsla.spsolve(IterASp,Iterrhs) 
				qqp_old = np.atleast_2d(q1_q2_p_new).T
			else:
				q1_q2_p_new = krypy.linsys.minres(IterASp, Iterrhs,
						#Ml = TsP.Ml, inner_product=ind2ra_ip,
						x0=qqp_old, maxiter=TsP.MaxIter, tol=TsP.linatol) 
				qqp_old = np.atleast_2d(q1_q2_p_new['xk'])

				q1_old = qqp_old[:Nv-(Np-1),]
				q2_old = qqp_old[Nv-(Np-1):Nv,]

				# Extract the 'actual' velocity and pressure
				vc = np.zeros((Nv,1))
				vc[~B2BoolInv,] = q1_old 
				vc[B2BoolInv,] = q2_old 
				pc = -1.0/dt*qqp_old[Nv:,]

				print 'Needed %d iterations -- prefinal relres = %e' %(len(q1_q2_p_new['relresvec']), q1_q2_p_new['relresvec'][-2] )

			v, p = expand_vp_dolfunc(PrP, vp=None, vc=vc, pc=pc, pdof = Pdof)
			
			tcur += dt

			# the errors and residuals 
			vCur, pCur = PrP.v, PrP.p 
			vCur.t = tcur
			pCur.t = tcur - dt

			ContiRes.append(comp_cont_error(v,FpbcSme,PrP.Q))
			VelEr.append(errornorm(vCur,v))
			PEr.append(errornorm(pCur,p))

		print '%d of %d time steps completed ' % (etap*Nts/TsP.NOutPutPts, Nts) 

		if TsP.ParaviewOutput:
			TsP.UpFiles.u_file << v, tcur
			TsP.UpFiles.p_file << p, tcur

	TsP.Residuals.ContiRes.append(ContiRes)
	TsP.Residuals.VelEr.append(VelEr)
	TsP.Residuals.PEr.append(PEr)
		
	return

def halfexp_euler_nseind2(Mc,Ac,BTc,Bc,fvbc,fpbc,vp_init,PrP,TsP):
	"""halfexplicit euler for the NSE in index 2 formulation
	"""

	Nts, t0, tE, dt, Nv, Np = init_time_stepping(PrP,TsP)

	tcur = t0

	v, p   = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None)
	TsP.UpFiles.u_file << v, tcur
	TsP.UpFiles.p_file << p, tcur

	# put -dt into the pressure for balancing and symmetry
	IterAv = sps.hstack([Mc,BTc[:,:-1]])
	IterAp = sps.hstack([Bc[:-1,:],sps.csr_matrix((Np-1,Np-1))])
	IterA  = sps.vstack([IterAv,IterAp])

	vp_old = vp_init
	ContiRes, VelEr, PEr = [], [], []
	
	def ind2_ip(vp1,vp2):
		"""to weight the conti eqn 

		for better comparability with ind1, 
		where there are two conti eqns
		"""
		v1, v2 = vp1[:Nv,], vp2[:Nv,]
		p1, p2 = vp1[Nv:,], vp2[Nv:,]
		return np.dot(v1.T.conj(),v2)+np.dot(p1.T.conj(),p2)

	for etap in range(1,TsP.NOutPutPts + 1 ):
		for i in range(Nts/TsP.NOutPutPts):

			ConV  = dtn.get_convvec(v, PrP.V)
			CurFv = dtn.get_curfv(PrP.V, PrP.fv, PrP.invinds, tcur)

			Iterrhs = np.vstack([Mc*vp_old[:Nv,],np.zeros((Np-1,1))]) \
					+ np.vstack([dt*(fvbc+CurFv-ConV[PrP.invinds,]),
						fpbc[:-1,]])

			if TsP.linatol == 0:
				vp_new = spsla.spsolve(IterA,Iterrhs)#,vp_old,tol=TsP.linatol)
				vp_old = np.atleast_2d(vp_new).T
			else:
				ret = krypy.linsys.minres(IterA, Iterrhs, x0=vp_old, tol=TsP.linatol)
				vp_old = ret['xk'] 
				print 'Needed %d iterations -- final relres = %e' %(len(ret['relresvec']), ret['relresvec'][-1] )

			vc = vp_old[:Nv,]
			pc = -1.0/dt*vp_old[Nv:,]

			v, p = expand_vp_dolfunc(PrP, vp=None, vc=vc, pc=pc)

			tcur += dt

			# the errors  
			vCur, pCur = PrP.v, PrP.p 
			vCur.t = tcur 
			pCur.t = tcur - dt

			ContiRes.append(comp_cont_error(v,fpbc,PrP.Q))
			VelEr.append(errornorm(vCur,v))
			PEr.append(errornorm(pCur,p))

		print '%d of %d time steps completed ' % (etap*Nts/TsP.NOutPutPts, Nts) 

		if TsP.ParaviewOutput:
			TsP.UpFiles.u_file << v, tcur
			TsP.UpFiles.p_file << p, tcur

	TsP.Residuals.ContiRes.append(ContiRes)
	TsP.Residuals.VelEr.append(VelEr)
	TsP.Residuals.PEr.append(PEr)
		
	return

def qr_impeuler(Mc,Ac,BTc,Bc,fvbc,fpbc,vp_init,PrP,TsP=None):
	""" with BTc[:-1,:] = Q*[R ; 0] 
	we define ~M = Q*M*Q.T , ~A = Q*A*Q.T , ~V = Q*v
	and condense the system accordingly """

	Nts, t0, tE, dt, Nv, Np = init_time_stepping(PrP,TsP)

	tcur = t0

	v, p   = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None)
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

	Tv1 = np.linalg.solve(Rm[:Np-1,].T, fpbc[:-1,])

	Tv2_old = np.dot(Qm_2, vp_init[:Nv,])

	Tfv2 = np.dot(Qm_2, fvbc) + np.dot(TA_21, Tv1)

	IterA2  = TM_22+dt*TA_22

	for etap in range(1,11):
		for i in range(Nts/10):
			tcur = tcur + dt

			Iter2rhs = np.dot(TM_22, Tv2_old) + dt*Tfv2
			Tv2_new = np.linalg.solve(IterA2, Iter2rhs)

			Tv2_old = Tv2_new

			# Retransformation v = Q.T*Tv
			vc_new = np.dot(Qm.T, np.vstack([Tv1, Tv2_new]))
			
			RhsTv2dot = Tfv2 - np.dot(TA_22, Tv2_new) 
			Tv2dot = np.linalg.solve(TM_22, RhsTv2dot)

			RhsP = np.dot(Qm_1,fvbc) - np.dot(TA_11,Tv1) \
					- np.dot(TA_12,Tv2_new) - np.dot(TM_12,Tv2dot)

			pc_new = - np.linalg.solve(Rm[:Np-1,],RhsP)

		v, p = expand_vp_dolfunc(PrP, vp=None, vc=vc_new, pc=pc_new)
		TsP.UpFiles.u_file << v, tcur
		TsP.UpFiles.p_file << p, tcur
		print '%d of %d time steps completed ' % (etap*Nts/10,Nts) 
		
	return

def plain_impeuler(Mc,Ac,BTc,Bc,fvbc,fpbc,vp_init,PrP,TsP):

	Nts, t0, tE, dt, Nv, Np = init_time_stepping(PrP,TsP)

	v, p   = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None)

	tcur = t0

	TsP.UpFiles.u_file << v, tcur
	TsP.UpFiles.p_file << p, tcur

	IterAv = sps.hstack([Mc+dt*Ac,-dt*BTc])
	IterAp = sps.hstack([-dt*Bc,sps.csr_matrix((Np,Np))])
	IterA  = sps.vstack([IterAv,IterAp]).todense()[:-1,:-1]

	vp_old = vp_init
	for etap in range(1,11):
		for i in range(Nts/10):
			tcur = tcur + dt

			Iterrhs = np.vstack([Mc*vp_old[:Nv,],np.zeros((Np-1,1))]) \
					+ dt*np.vstack([fvbc,fpbc[:-1,]])
			vp_new = np.linalg.solve(IterA,Iterrhs)
			vp_old = vp_new


		print '%d of %d time steps completed ' % (etap*Nts/10,Nts) 
		v, p = expand_vp_dolfunc(PrP, vp=vp_new, vc=None, pc=None)

		TsP.UpFiles.u_file << v, tcur
		TsP.UpFiles.p_file << p, tcur
		
	return

def comp_cont_error(v,fpbc,Q):
	"""Compute the L2 norm of the residual of the continuity equation
		Bv = g
	"""

	q = TrialFunction(Q)
	divv = assemble(q*div(v)*dx)

	conRhs = Function(Q)
	conRhs.vector().set_local(fpbc)

	#raise Warning('debugggg')

	ContEr = norm(conRhs.vector()-divv)

	return ContEr


def expand_vp_dolfunc(PrP, vp=None, vc=None, pc=None, pdof=None):
	"""expand v and p to the dolfin function representation
	
	pdof = pressure dof that was set zero
	"""

	v = Function(PrP.V)
	p = Function(PrP.Q)

	if vp is not None:
		if vp.ndim == 1:
			vc = vp[:len(PrP.invinds)].reshape(len(PrP.invinds),1)
			pc = vp[len(PrP.invinds):].reshape(PrP.Q.dim()-1,1)
		else:
			vc = vp[:len(PrP.invinds),:]
			pc = vp[len(PrP.invinds):,:]

	ve = np.zeros((PrP.V.dim(),1))

	# fill in the boundary values
	for bc in PrP.velbcs:
		bcdict = bc.get_boundary_values()
		ve[bcdict.keys(),0] = bcdict.values()

	ve[PrP.invinds] = vc

	if pdof is None:
		pe = np.vstack([pc,[0]])
	elif pdof == 0:
		pe = np.vstack([[0],pc])
	elif pdof == -1:
		pe = pc
	else:
		pe = np.vstack([pc[:pdof],np.vstack([[0.02],pc[pdof:]])])

	v.vector().set_local(ve)
	p.vector().set_local(pe)

	return v, p

def get_conv_curfv_rearr(v,PrP,tcur,B2BoolInv):

		ConV  = dtn.get_convvec(v, PrP.V)
		ConV = ConV[PrP.invinds,]

		ConV = np.vstack([ConV[~B2BoolInv],ConV[B2BoolInv]])

		CurFv = dtn.get_curfv(PrP.V, PrP.fv, PrP.invinds, tcur)
		if len(CurFv) != len(PrP.invinds):
			raise Warning('Need fv at innernodes here')
		CurFv = np.vstack([CurFv[~B2BoolInv],CurFv[B2BoolInv]])

		return ConV, CurFv


def init_time_stepping(PrP,TsP):
	"""what every method starts with """
	
	Nts, t0, tE = TsP.Nts, TsP.t0, TsP.tE
	dt = (tE-t0)/Nts
	Nv = len(PrP.invinds)
	Np = PrP.Q.dim()

	if Nts % TsP.NOutPutPts != 0:
		TsP.NOutPutPts = 1

	return Nts, t0, tE, dt, Nv, Np
