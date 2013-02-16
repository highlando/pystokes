import numpy as np
from dolfin import *

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

