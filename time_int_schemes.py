from dolfin import *
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import krypy.linsys
from scipy.linalg import qr

import dolfin_to_nparrays as dtn

###
# solve M\dot v + Av -B'p = fv
#                 Bv      = fpbc
###

def halfexp_euler_smarminex(MSme,BSme,FvbcSme,FpbcSme,vp_init,B2BoolInv,PrP,TsP):
	"""halfexplicit euler for the NSE in index 1 formulation
	
	"""

	N, Pdof = PrP.N, PrP.Pdof

	Nts, t0, tE, dt, Nv, Np = init_time_stepping(PrP,TsP)
	tcur = t0

	# remove the p - freedom
	if Pdof == 0:
		BSme  = BSme[1:,:]
		FpbcSme = FpbcSme[1:,]
	else:
		BSme  = sps.vstack([BSme[:Pdof,:],BSme[Pdof+1:,:]])

	B1Sme = BSme[:,:Nv-(Np-1)]
	B2Sme = BSme[:,Nv-(Np-1):]

	M1Sme = MSme[:,:Nv-(Np-1)]
	M2Sme = MSme[:,Nv-(Np-1):]
	
	#### The matrix to be solved in every time step
	#
	# 		M11    M12  -dt*B2'  0          q1
	# 		M21    M22  -dt*B1'  0       dt*tq2
	# 		B1     B2      0     0   *      p     = rhs
	# 		B1     0       0     B2 	    q2  
	# 						    		 
	# cf. preprint
	#


	IterA1 = sps.hstack([MSme,-dt*BSme.T])

	# Multiply by -dt for symmetry
	IterA2 = sps.hstack([-dt*BSme, sps.csr_matrix((Np-1, Np-1))])

	IterASp = sps.vstack([IterA1,IterA2])
	
	IterA3 = sps.hstack([sps.hstack([B1Sme,sps.csr_matrix((Np-1,2*(Np-1)))]),
		B2Sme])

	IterA = sps.vstack([
		sps.hstack([IterASp, sps.csr_matrix((Nv+Np-1,Np-1))]),
				IterA3])

	v, p   = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None, pdof=None)
	TsP.UpFiles.u_file << v, tcur
	TsP.UpFiles.p_file << p, tcur

	vp_old = np.copy(vp_init)
	q1_old = vp_init[~B2BoolInv,]
	q2_old = vp_init[B2BoolInv,]

	# state vector of the smaminex system : [ q1^+, tq2^c, p^c, q2^+] 
	qqpq_old = np.zeros((Nv+Np-1,1))
	qqpq_old[:Nv-(Np-1),] = q1_old
	qqpq_old[Nv:Nv+Np-1,] = vp_old[Nv:,]
	qqpq_old[Nv+Np-1:,] = q2_old[Nv:,]

	qqp_old = qqpq_old[:Nv+Np-1,]

	ContiRes, VelEr, PEr = [], [], []

	for etap in range(1,TsP.SampInt +1 ):
		for i in range(Nts/TsP.SampInt):
			ConV, CurFv = get_conv_curfv_rearr(v,PrP,tcur,B2BoolInv)

			gdot = np.zeros((Np-1,1)) # TODO: implement \dot g

			Iterrhs = np.vstack([M1Sme*q1_old, -dt*B1Sme*q1_old]) \
						+ dt*np.vstack([FvbcSme+CurFv-ConV,gdot])

			if TsP.Split:
				if TsP.linatol == 0:
					q1_tq2_p_new = spsla.spsolve(IterASp,Iterrhs) 
					qqp_old = np.atleast_2d(q1_tq2_p_new).T
				else:
					q1_tq2_p_new = krypy.linsys.minres(IterASp, Iterrhs,
							x0=qqp_old, tol=TsP.linatol) 
					qqp_old = np.atleast_2d(q1_tq2_p_new['xk'])

				q1_old = qqp_old[:Nv-(Np-1),]
				q2_old = spsla.spsolve(B2Sme, FpbcSme - B1Sme*q1_old) 
				q2_old = np.atleast_2d(q2_old).T

				qSmaMin = np.vstack([q1_old, q2_old])
				# Extract the 'actual' velocity and pressure
				vc = np.zeros((Nv,1))
				vc[~B2BoolInv,] = q1_old 
				vc[B2BoolInv,] = q2_old 
				pc = qqp_old[Nv:,]

			else:
				Iterrhs = np.vstack([Iterrhs,FpbcSme])
				if TsP.linatol == 0:
					q1_tq2_p_q2_new = spsla.spsolve(IterA,Iterrhs) 
					qqpq_old = np.atleast_2d(q1_tq2_p_q2_new).T
				else:
					q1_tq2_p_q2_new = krypy.linsys.gmres(IterA, Iterrhs,
							x0=qqpq_old, Ml=TsP.Ml, Mr=TsP.Mr, 
							tol=TsP.linatol, max_restarts=0, maxiter=500)
					qqpq_old = np.atleast_2d(q1_tq2_p_q2_new['xk'])

				# Extract the 'actual' velocity and pressure
				vc = np.zeros((Nv,1))
				vc[~B2BoolInv,] = qqpq_old[:Nv-(Np-1),]
				vc[B2BoolInv,] = qqpq_old[-(Np-1):,]
				pc = qqpq_old[Nv:Nv+Np-1,]

			v, p = expand_vp_dolfunc(PrP, vp=None, vc=vc, pc=pc, pdof = Pdof)
			
			tcur += dt

		# the errors  
		vCur, pCur = PrP.v, PrP.p 
		vCur.t = tcur
		pCur.t = tcur - dt

		print '%d of %d time steps completed ' % (etap*Nts/TsP.SampInt, Nts) 

		if TsP.ParaviewOutput:
			TsP.UpFiles.u_file << v, tcur
			TsP.UpFiles.p_file << p, tcur

		ContiRes.append(comp_cont_error(v,FpbcSme,PrP.Q))
		VelEr.append(errornorm(vCur,v))
		PEr.append(errornorm(pCur,p))

	TsP.Residuals.ContiRes.append(ContiRes)
	TsP.Residuals.VelEr.append(VelEr)
	TsP.Residuals.PEr.append(PEr)
		
	return


def halfexp_euler_smarminex_wminresprec(Mc,Ac,BTc,Bc,fvbc,fpbc,vp_init,B2BoolInv,PrP,TsP):
	"""halfexplicit euler for the NSE in index 2 formulation

	iterative scheme: we first solve the symmetric subproblem
	and use this solution as starting solution for gmres
	"""

	Nts, t0, tE, dt, Nv, Np = init_time_stepping(PrP,TsP)
	tcur = t0

	# Sort and flatten the B2BoolInv
	#B2BoolInv = np.sort(B2BoolInv, axis=None).astype(int)

	# the complement of the bubble index in the inner nodes
	BubIndC = ~B2BoolInv #np.setdiff1d(np.arange(Nv),B2BoolInv)
	# the bubbles as indices
	B2BI = np.arange(len(B2BoolInv))[B2BoolInv]

	# Reorder the matrices for smart min ext
	MSme = col_columns_atend(Mc, B2BI)
	ASme = col_columns_atend(Ac, B2BI)
	BSme = col_columns_atend(Bc, B2BI)

	BTSme = BSme.T

	# here the first pressure dof is set zero
	B1Sme = BSme[1:,:][:,:Nv-(Np-1)]
	B2Sme = BSme[1:,:][:,Nv-(Np-1):]

	M1Sme = MSme[:,:Nv-(Np-1)]
	M2Sme = MSme[:,Nv-(Np-1):]
	
	#### The matrix to be solved in every time step
	#
	# 		M1  dt*M2  -dt*B'  0      q1
	# 		B1  dt*B2   0      0   *  tq2  = rhs
	# 		B1    0     0     B2 	  p
	# 								  q2
	# cf. preprint

	IterA1 = sps.hstack([sps.hstack([M1Sme,dt*M2Sme]),
		sps.hstack([-dt*BTc[:,1:],sps.csr_matrix((Nv,Np-1))])])

	IterA2 = sps.hstack([sps.hstack([B1Sme,dt*B2Sme]),
		sps.csr_matrix((Np-1, 2*(Np-1)))])
	
	IterA3 = sps.hstack([sps.hstack([B1Sme,sps.csr_matrix((Np-1,2*(Np-1)))]),
		B2Sme])

	IterA = sps.vstack([sps.vstack([IterA1,IterA2]),IterA3])

	# symmetric subproblem
	IterAqq = sps.hstack([Mc,-dt*BTc[:,1:]])
	IterAp = sps.hstack([-dt*Bc[1:,:],sps.csr_matrix((Np-1,Np-1))])
	# multiplied by -dt for symmetry

	IterAMinR  = sps.vstack([IterAqq,IterAp])
	
	v, p   = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None, pdof=None)
	TsP.UpFiles.u_file << v, tcur
	TsP.UpFiles.p_file << p, tcur

	vp_old = vp_init
	q1_old = vp_init[BubIndC,]
	# state vector of the smaminex system : [ q1^+, tq2^c, p^c, q2^+], cf. preprint
	qqpq_old = np.zeros((Nv+2*(Np-1),1))
	qqpq_old[:Nv-(Np-1),] = q1_old
	qqpq_old[Nv:Nv+Np-1,] = vp_old[Nv:,]
	qqpq_old[Nv+Np-1:]    = vp_old[B2BoolInv,]

	# state vector of the smaminex system : [ q1^+, tq2^c, p^c] with [q2^+] decoupled
	qqp_old = np.zeros((Nv+Np-1,1))
	qqp_old[:Nv-(Np-1),] = q1_old
	qqp_old[Nv:Nv+Np-1,] = vp_old[Nv:,]

	ContiRes, VelEr, PEr = [], [], []

	Mr = sps.spdiags(np.r_[np.ones(Nv-(Np-1)), 1/dt*np.ones(Np-1), np.ones(2*(Np-1))],0,Nv+2*(Np-1),Nv+2*(Np-1))
	Ml = sps.spdiags(np.r_[np.ones(Nv-(Np-1)), np.ones(Np-1), dt*np.ones(2*(Np-1))],0,Nv+2*(Np-1),Nv+2*(Np-1))
	for etap in range(1,11):
		for i in range(Nts/8):

			ConV  = dtn.get_convvec(v, PrP.V)
			CurFv = dtn.get_curfv(PrP.V, PrP.fv, PrP.invinds, tcur)

			Iterrhs = np.vstack([M1Sme*q1_old,
				np.vstack([B1Sme*q1_old,fpbc[1:,]])]) \
						+ dt*np.vstack([fvbc+CurFv-Ac*vp_old[:Nv,]-ConV[PrP.invinds,],
						np.zeros((2*(Np-1),1))])

			gdot = np.zeros((Np-1,1)) # TODO: implement \dot g

			ItrhsMinR = np.vstack([M1Sme*q1_old, -dt*B1Sme*q1_old]) \
						+ dt*np.vstack([fvbc+CurFv-0*Ac*vp_old[:Nv,]-ConV[PrP.invinds,],gdot])

			if TsP.linatol == 0:
				raise Warning('No direct solve for this method')
			else:
				tic = time.clock()

				GetStartByMinres = False
				## Solve the sym subprob
				if GetStartByMinres: 
					q1_tq2_p_new = krypy.linsys.minres(IterAMinR, ItrhsMinR,
							x0=qqp_old, tol=TsP.linatol) 
					qqp_old = np.atleast_2d(q1_tq2_p_new['xk'])

					q1_old  = qqp_old[BubIndC,]
					tq2_old = 1/dt*qqp_old[B2BoolInv,] #the dt was shifted into tq2
					p_old   = qqp_old[Nv:,]

					qqpq_old[:Nv-(Np-1),] = q1_old
					qqpq_old[Nv:Nv+Np-1,] = p_old
					qqpq_old[Nv-(Np-1):Nv,]= tq2_old


				q1_tq2_p_q2_new = krypy.linsys.gmres(IterA, Iterrhs,
						x0=qqpq_old, Ml=Ml, Mr=Mr, 
						tol=TsP.linatol, max_restarts=0, maxiter=500)

				toc = time.clock()
				print toc - tic

				qqpq_old = np.atleast_2d(q1_tq2_p_q2_new['xk'])

			# Extract the 'actual' velocity and pressure
			vcSmaMin = np.vstack([qqpq_old[:Nv-(Np-1),],
								  qqpq_old[-(Np-1):,]])

			vc = revert_sort_tob2(vcSmaMin,B2BI)
			pc = qqpq_old[Nv:Nv+Np-1,]

			vp_old = np.vstack([vc,pc])


			v_old1 = qqpq_old[:Nv-(Np-1),]
			
			v, p = expand_vp_dolfunc(PrP, vp=None, vc=vc, pc=pc, pdof = 0)
			
			tcur += dt

		# the errors  
		vCur, pCur = PrP.v, PrP.p 
		vCur.t = tcur
		pCur.t = tcur - dt

		print '%d of %d time steps completed ' % (etap*Nts/10,Nts) 

		TsP.UpFiles.u_file << v, tcur
		TsP.UpFiles.p_file << p, tcur

		ContiRes.append(comp_cont_error(v,fpbc,PrP.Q))
		VelEr.append(errornorm(vCur,v))
		PEr.append(errornorm(pCur,p))

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

	IterAv = sps.hstack([Mc+0*dt*Ac,-dt*BTc[:,:-1]])
	#-dt*Bc = conti mult. by -dt, to make it symmetric for using minres
	IterAp = sps.hstack([-dt*Bc[:-1,:],sps.csr_matrix((Np-1,Np-1))])
	IterA  = sps.vstack([IterAv,IterAp])

	vp_old = vp_init
	ContiRes, VelEr, PEr = [], [], []
	for etap in range(1,TsP.SampInt + 1 ):
		for i in range(Nts/TsP.SampInt):

			#vp_old[Nv:,0] = 0 

			ConV  = dtn.get_convvec(v, PrP.V)
			CurFv = dtn.get_curfv(PrP.V, PrP.fv, PrP.invinds, tcur)

			Iterrhs = np.vstack([Mc*vp_old[:Nv,],np.zeros((Np-1,1))]) \
					+ np.vstack([dt*(fvbc+CurFv-0*Ac*vp_old[:Nv,]-ConV[PrP.invinds,]),
						-dt*fpbc[:-1,]])

			if TsP.linatol == 0:
				vp_new = spsla.spsolve(IterA,Iterrhs)#,vp_old,tol=TsP.linatol)
				vp_old = np.atleast_2d(vp_new).T
			else:
				ret = krypy.linsys.minres(IterA, Iterrhs, x0=vp_old, tol=TsP.linatol)
				vp_old = ret['xk'] 
			
			v, p = expand_vp_dolfunc(PrP, vp=vp_old, vc=None, pc=None)

			tcur += dt

			# if TsP.UseRealPress:
			# 	PrP.p.t = tcur - dt
			# 	pexa = project(PrP.p, PrP.Q)
			# 	pexa = pexa.vector()
			# 	pexa = pexa.array()[1:,]
			

		# the errors  
		vCur, pCur = PrP.v, PrP.p 
		vCur.t = tcur 
		pCur.t = tcur - dt

		print '%d of %d time steps completed ' % (etap*Nts/TsP.SampInt, Nts) 

		if TsP.ParaviewOutput:
			TsP.UpFiles.u_file << v, tcur
			TsP.UpFiles.p_file << p, tcur

		ContiRes.append(comp_cont_error(v,fpbc,PrP.Q))
		VelEr.append(errornorm(vCur,v))
		PEr.append(errornorm(pCur,p))

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

	return Nts, t0, tE, dt, Nv, Np
