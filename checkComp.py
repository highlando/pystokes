import numpy as np
import scipy.sparse as sps

import test_time_schemes as tts
import time_int_schemes as tis
reload(tis)

import dolfin_to_nparrays as dtn
reload(dtn)

import smartminex_tayhoomesh as smt
reload(smt)

N = 32 

PrP = tts.ProbParams(N)

Bub2 = smt.get_B2_bubbleinds(N,PrP.V,PrP.mesh,Q=PrP.V).astype(int)
B2BubBool = np.in1d(np.arange(PrP.V.dim())[PrP.invinds], Bub2)
BubI2 = np.arange(len(B2BubBool))[B2BubBool]

Ma, Aa, BTa, Ba = dtn.get_sysNSmats(PrP.V, PrP.Q)
fv, fp = dtn.setget_rhs(PrP.V, PrP.Q, PrP.fv, PrP.fp)
Mc, Ac, BTc, Bc, fvbc, fpbc, bcinds, bcvals, invinds = \
		dtn.condense_sysmatsbybcs(Ma,Aa,BTa,Ba,fv,fp,PrP.velbcs)

BcC = sps.csr_matrix(Bc, copy=True)

BSme = tis.col_columns_atend(BcC, BubI2)
B2Sme = BSme[1:,:][:,-len(Bub2):]

print np.linalg.cond(B2Sme.todense())

# np.set_printoptions(precision=2)
#print B2Sme.todense()
# print BubI2 
# print Bc[1:,:][:,:].todense()
