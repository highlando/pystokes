from dolfin import *
from scipy.sparse import csr_matrix
import numpy as np

parameters.linear_algebra_backend = "uBLAS"

mesh = UnitSquareMesh(4, 4)

# Define mixed FEM function spaces
V = VectorFunctionSpace(mesh, "CG", 2)

v = Function(V)
ve = np.ones((V.dim(),1))
v.vector().set_local(ve)
plot(v)

u_file = File("velocitty.pvd")
u_file << v


