from dolfin import *
import smartminex_tayhoomesh 
import numpy as np

N = 2

mesh = smartminex_tayhoomesh.getmake_mesh(N)

V = VectorFunctionSpace(mesh, "CG", 2)

# We sort out the bubble functions - dofs on edge midpoints 
BubDofs = np.zeros((3*4*(N-1)**2,4))
# This will be the array of 
# [x y dofx dofy]

for (i, cell) in enumerate(cells(mesh)):
	print "Global dofs associated with cell %d: " % i,
	print V.dofmap().cell_dofs(i)
	Cdofs = V.dofmap().cell_dofs(i)
	print "The Dof coordinates:",
	print V.dofmap().tabulate_coordinates(cell)
	Coos = V.dofmap().tabulate_coordinates(cell)

	# In every cell the bubbles are numbered 4th-6th (x)
	# and 10th-12th (y-comp)
	CelBubDofs = np.vstack([Cdofs[3:6],Cdofs[9:12]]).T
	CelBubCoos = Coos[3:6]

	BubDofs[i*3:(i+1)*3,:] = np.hstack([CelBubCoos,CelBubDofs]) 

# remove duplicate entries 
# actually one can remove them completely == inner edge bubbles
aux, indics = np.unique(BubDofs[:,-1], return_index = True)
BubDofsUnq = BubDofs[indics, :]

### identify the bubbles of choice


