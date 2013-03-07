from dolfin import project
import test_time_schemes as tts
import dolfin_to_nparrays as dtn
import numpy as np

N = 32 
tcur = 0.5
nu   = 0

PrP = tts.ProbParams(N)
# get system matrices as np.arrays
Ma, Aa, BTa, Ba = dtn.get_sysNSmats(PrP.V, PrP.Q)

PrP.fv.t  = tcur
PrP.fv.nu = nu   # nu = 0, for Eulerian Flow - no diffusion

fv, fp = dtn.setget_rhs(PrP.V, PrP.Q, PrP.fv, PrP.fp)

Mc, Ac, BTc, Bc, fvbc, fpbc, bcinds, bcvals, invinds = \
		dtn.condense_sysmatsbybcs(Ma,Aa,BTa,Ba,fv,fp,PrP.velbcs)

v, vdot, p = PrP.v, PrP.vdot, PrP.p

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

vc = get_funcexpr_as_vec(v, PrP.V, t=tcur)
pc = get_funcexpr_as_vec(p, PrP.Q, t=tcur)

vdotc = get_funcexpr_as_vec(vdot, PrP.V, t=tcur)

v.t = tcur
vf = project(v, PrP.V)
ConV  = dtn.get_convvec(vf, PrP.V)

resV = Ma*vdotc + nu*Aa*vc + ConV[:,0] - BTa*pc - fv[:,0]
resVc = Mc*vdotc[invinds] + nu*Ac*vc[invinds] + ConV[invinds,0] - BTc*pc - fvbc[:,0] - fv[invinds,0]

print np.dot(np.atleast_2d(resV[invinds]),Mc*np.atleast_2d(resV[invinds]).T)
print np.dot(np.atleast_2d(resVc),Mc*np.atleast_2d(resVc).T)
