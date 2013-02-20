from dolfin import *

mesh = UnitSquare(2,2)
V = FunctionSpace(mesh, "CG", 1)

for (i, cell) in enumerate(cells(mesh)):
	print "Global dofs associated with cell %d: " % i,
	print V.dofmap().cell_dofs(i)
	print "Dof the coordinates:",
	print V.dofmap().tabulate_coordinates(cell)
