import numpy as np
from astropy.io import fits
import sys


def ind2xy(ind, x, y):
	indx = ind%x
	indy = np.floor(ind/x)
	return indx, indy

def alter_loc_flag(i, L):

	if (L%2) == 0:                 ## even number
		if i<=(L/2-1): 
			flag=1
		elif i>=(L/2): 
			flag=-1
	else:                          ## odd number
		if i<(L-1)/2: 
			flag=1
		elif i==(L-1)/2: 
			flag=0
		elif i>(L-1)/2: 
			flag=-1

	return flag

def peri_dist(ix, iy, jxs, jys, x, y):
	
	xflag = alter_loc_flag(ix, x)
	yflag = alter_loc_flag(iy, y)

	dist_min = np.sqrt( (ix-jxs)**2 + (iy-jys)**2)

	if xflag !=0:
		dist_alter_x = np.sqrt( (ix + x*xflag -jxs)**2 + (iy-jys)**2)
		dist_min = np.minimum(dist_min, dist_alter_x)
	if yflag != 0:
		dist_alter_y = np.sqrt( (ix-jxs)**2 + (iy + y*yflag -jys)**2)
		dist_min = np.minimum(dist_min, dist_alter_y)

	if (xflag !=0) & (yflag !=0):
		dist_alter_xy =  np.sqrt( (ix + x*xflag -jxs)**2 + (iy + y*yflag -jys)**2)
		dist_min = np.minimum(dist_min, dist_alter_xy)

	return dist_min

###############################################

path = './distLib/'

x = int(sys.argv[1])
y = int(sys.argv[2])
boundary = sys.argv[3]

if boundary not in ['Periodic', 'Dirichlet']:
	raise Exception('Boundary is not Periodic or Dirichlet')

npix = x*y
print("number of pixels: ", npix)

distLib = np.zeros((npix, npix))


if npix%2 == 0:

	for i in range(int(npix/2)):

		ix, iy = ind2xy(i, x, y)
		if i%100 == 0:
			print( "calculating distance library:", i, "/", npix)

		js = np.array(range(i+1, npix))
		jxs, jys = ind2xy(js, x, y)

		if boundary == 'Dirichlet':
			dist = np.sqrt( (ix-jxs)**2 + (iy-jys)**2)
		elif boundary == 'Periodic':
			dist = peri_dist(ix, iy, jxs, jys, x, y)

		distLib[i, i+1:] = dist

	for i in range(int(npix/2)):
		distLib[:,i] = distLib[i,:]

	for i in range(int(npix/2), npix-1):
		distLib[i, i+1:] = np.flip(distLib[npix-i-1, :npix-i-1])

	for i in range(int(npix/2), npix-1):
		distLib[:,i] = distLib[i,:]

else:

	for i in range(int((npix+1)/2)):

		ix, iy = ind2xy(i, x, y)
		if i%100 == 0:
			print( "calculating distance library:", i, "/", npix)

		js = np.array(range(i+1, npix))
		jxs, jys = ind2xy(js, x, y)

		if boundary == 'Dirichlet':
			dist = np.sqrt( (ix-jxs)**2 + (iy-jys)**2)
		elif boundary == 'Periodic':
			dist = peri_dist(ix, iy, jxs, jys, x, y)

		distLib[i, i+1:] = dist

	for i in range(int((npix+1)/2)):
		distLib[:,i] = distLib[i,:]

	for i in range(int((npix+1)/2), npix-1):
		distLib[i, i+1:] = np.flip(distLib[npix-i-1, :npix-i-1])

	for i in range(int((npix+1)/2), npix-1):
		distLib[:,i] = distLib[i,:]

hdu = fits.PrimaryHDU(distLib)
hdu.header['x'] = x
hdu.header['y'] = y
hdu.header['npix'] = npix

hdu.writeto(path + f'distLib{x}x{y}_{boundary}.fits', overwrite=True)
print('Distance Matrix saved as:', path + f'distLib{x}x{y}_{boundary}.fits')

