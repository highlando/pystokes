import numpy as np
from dolfin import Mesh, cells

def get_smamin_rearrangement(N,PrP,Mc,Bc):
	from smamin_utils import col_columns_atend
	from scipy.io import loadmat, savemat

	# get the indices of the B2-part
	B2Inds = get_B2_bubbleinds(N, PrP.V, PrP.mesh)
	# the B2 inds wrt to inner nodes
	# this gives a masked array of boolean type
	B2BoolInv = np.in1d(np.arange(PrP.V.dim())[PrP.invinds], B2Inds)
	# this as indices
	B2BI = np.arange(len(B2BoolInv), dtype=np.int32)[B2BoolInv]

	dname = '%sSmeMcBc' % N

	try: SmDic = loadmat(dname)

	except IOError:
		print 'Computing the B2 indices...'
		# get the indices of the B2-part
		B2Inds = get_B2_bubbleinds(N, PrP.V, PrP.mesh)
		# the B2 inds wrt to inner nodes
		# this gives a masked array of boolean type
		B2BoolInv = np.in1d(np.arange(PrP.V.dim())[PrP.invinds], B2Inds)
		# this as indices
		B2BI = np.arange(len(B2BoolInv), dtype=np.int32)[B2BoolInv]
		# Reorder the matrices for smart min ext...
		# ...the columns
		print 'Rearranging the matrices...'
		# Reorder the matrices for smart min ext...
		# ...the columns
		MSmeC = col_columns_atend(Mc, B2BI)
		BSme = col_columns_atend(Bc, B2BI)
		# ...and the lines
		MSmeCL = col_columns_atend(MSmeC.T, B2BI)
		print 'done'

		savemat(dname, { 'MSmeCL': MSmeCL, 'BSme':BSme, 
			'B2Inds':B2Inds, 'B2BoolInv':B2BoolInv, 'B2BI':B2BI} )
	
	SmDic = loadmat(dname)

	MSmeCL = SmDic['MSmeCL']
	BSme = SmDic['BSme']
	B2Inds = SmDic['B2Inds']
	B2BoolInv = SmDic['B2BoolInv']>0
	B2BoolInv = B2BoolInv.flatten()
	B2BI = SmDic['B2BI']

	return MSmeCL, BSme, B2Inds, B2BoolInv, B2BI 

def getmake_mesh(N):
	"""write the mesh for the smart minext tayHood square

	order is I. main grid, II. subgrid = grid of the cluster centers
	and in I and II lexikographical order
	first y-dir, then x-dir """

	try:
		f = open('smegrid%s.xml' % N)
	except IOError:
		print 'Need generate the mesh...'

		# main grid
		h = 1./(N-1)
		y, x = np.ogrid[0:N,0:N]
		y = h*y+0*x
		x = h*x+0*y
		mgrid = np.hstack((y.reshape(N**2,1), x.reshape(N**2,1)))

		# sub grid
		y, x = np.ogrid[0:N-1,0:N-1]
		y = h*y+0*x
		x = h*x+0*y
		sgrid = np.hstack((y.reshape((N-1)**2,1), x.reshape((N-1)**2,1)))

		grid = np.vstack((mgrid,sgrid+0.5*h))

		f = open('smegrid%s.xml' % N, 'w')
		f.write('<?xml version="1.0"?> \n <dolfin xmlns:dolfin="http://www.fenicsproject.org"> \n <mesh celltype="triangle" dim="2"> \n')

		f.write('<vertices size="%s">\n' % (N**2+(N-1)**2) )
		for k in range(N**2+(N-1)**2):
			f.write('<vertex index="%s" x="%s" y="%s" />\n' % (k, grid[k,0], grid[k,1]))
		
		f.write('</vertices>\n')
		f.write('<cells size="%s">\n' % (4*(N-1)**2))
		for j in range(N-1):
			for i in range(N-1):
				# number of current cluster center
				k = j*(N-1) + i 
				# vertices of the main grid in the cluster
				v0, v1, v2, v3 = j*N+i, (j+1)*N+i, (j+1)*N+i+1, j*N+i+1 

				f.write('<triangle index="%s" v0="%s" v1="%s" v2="%s" />\n' % (4*k,   v0, N**2+k, v1))
				f.write('<triangle index="%s" v0="%s" v1="%s" v2="%s" />\n' % (4*k+1, v1, N**2+k, v2))
				f.write('<triangle index="%s" v0="%s" v1="%s" v2="%s" />\n' % (4*k+2, v2, N**2+k, v3)) 
				f.write('<triangle index="%s" v0="%s" v1="%s" v2="%s" />\n' % (4*k+3, v3, N**2+k, v0)) 

		f.write('</cells>\n')
		
		f.write('</mesh> \n </dolfin> \n')
		f.close()

		print 'done'

	mesh = Mesh('smegrid%s.xml' % N)

	return mesh

def get_ij_subgrid(k,N):
	"""to get i,j numbering of the cluster centers of smaminext"""

	n = N-1
	if k > n**2-1 or k < 0:
		raise Exception('%s: No such node on the subgrid!' % k)
		
	j = np.mod(k,n)
	i = (k-j)/n
	return j, i

def get_B2_bubbleinds(N, V, mesh, Q=None):
	"""compute the indices of bubbels that set up

	the invertible B2. This function is specific for the 
	mesh generated by smartmintex_tayhoomesh ..."""

	# mesh V must be from
	# mesh = smartminex_tayhoomesh.getmake_mesh(N)
	# V = VectorFunctionSpace(mesh, "CG", 2)

	# 3 bubs * 4 cells * (N-1)**2 cluster
	BubDofs = np.zeros((3*4*(N-1)**2,4))
	# This will be the array of 
	# [x y dofx dofy]

	if Q is None:
		Q = V

	for (i, cell) in enumerate(cells(mesh)):
		# print "Global dofs associated with cell %d: " % i,
		# print Q.dofmap().cell_dofs(i)
		# print "The Dof coordinates:",
		# print Q.dofmap().tabulate_coordinates(cell)
		Cdofs = V.dofmap().cell_dofs(i)
		Coos = V.dofmap().tabulate_coordinates(cell)

		# We sort out the bubble functions - dofs on edge midpoints 
		# In every cell the bubbles are numbered 4th-6th (x)
		# and 10th-12th (y-comp)
		CelBubDofs = np.vstack([Cdofs[9:12],Cdofs[3:6]]).T
		CelBubCoos = Coos[3:6]

		BubDofs[i*3:(i+1)*3,:] = np.hstack([CelBubCoos,CelBubDofs]) 

	# remove duplicate entries
	yDofs = BubDofs[:,-1]
	Aux, IndBubToKeep = np.unique(yDofs, return_index = True)
	BubDofs = BubDofs[IndBubToKeep,:]

	# remove bubbles at cluster boarders
	# x
	XCors = BubDofs[:,0]
	XCors = np.rint(4*(N-1)*XCors)
	IndBorBub = np.in1d(XCors,np.arange(0,4*(N-1)+1,4))
	BubDofs = BubDofs[~IndBorBub,:]
	# y
	YCors = BubDofs[:,1]
	YCors = np.rint(4*(N-1)*YCors)
	IndBorBub = np.in1d(YCors,np.arange(0,4*(N-1)+1,4))
	BubDofs = BubDofs[~IndBorBub,:]

	# sort by y coordinate
	BubDofs = BubDofs[BubDofs[:,1].argsort(kind='mergesort')]
	# and by x !!! necessarily by mergesort
	BubDofs = BubDofs[BubDofs[:,0].argsort(kind='mergesort')]
	# no we have lexicographical order first y then x


	### identify the bubbles of choice
	BD = BubDofs

	VelBubsChoice = np.zeros(0,)
	CI = 2*(N-1) # column increment
	# First column of Cluster
	# First cluster
	ClusCont = np.array([BD[0,3],BD[1,2],BD[CI,2],BD[CI+1,3]])
	VelBubsChoice = np.append(VelBubsChoice,ClusCont)

	# loop over the rows
	for iCR in range(1,N-1):
		ClusCont = np.array([
			BD[2*iCR+1   ,2],
			BD[2*iCR+CI  ,2],
			BD[2*iCR+CI+1,3]])
		VelBubsChoice = np.append(VelBubsChoice,ClusCont)

	# loop over the columns
	for jCR in range(1,N-1):
		CC = (2*jCR)*2*(N-1) #current column
		# first cluster separate
		ClusCont = np.array([
			BD[CC,     3],
			BD[CC+CI,  2],
			BD[CC+CI+1,3]])
		VelBubsChoice = np.append(VelBubsChoice,ClusCont)

		# loop over the rows
		for iCR in range(1,N-1):
			ClusCont = np.array([
				BD[CC+2*iCR+CI  ,2],
				BD[CC+2*iCR+CI+1,3]])
			VelBubsChoice = np.append(VelBubsChoice,ClusCont)

	return VelBubsChoice.astype(int)
