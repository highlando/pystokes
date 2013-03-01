from dolfin import *
import smartminex_tayhoomesh 

#mesh = UnitSquare(2,2)
mesh = smartminex_tayhoomesh.getmake_mesh(3)

V = VectorFunctionSpace(mesh, "CG", 1)

for (i, cell) in enumerate(cells(mesh)):
	print "Global dofs associated with cell %d: " % i,
	print V.dofmap().cell_dofs(i)
	print "The Dof coordinates:",
	print V.dofmap().tabulate_coordinates(cell)
