"""

some different ways to define bulges of galaxies

"""




def measure_bulge_mass(x,y,vx,vy,mass):
    rpos = np.sqrt(x*x + y*y)
    vtan = (x*vy- y*vx)/rpos
    # define prograde
    direction = np.sign(np.nanmean(vtan))
    vtan *= direction
    negvel = np.where(vtan<0)[0]
    print('50% bulge: {0:3.2}kpc'.format(np.nanpercentile(rpos[negvel],50.)))
    return 2.*np.nansum(mass[negvel])
