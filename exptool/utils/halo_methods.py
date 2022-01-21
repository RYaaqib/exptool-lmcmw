#########################################################
#
#  Utility routines for Spherical expansion
#
#  MSP 5.1.2016
#
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy import interpolate

# pull in exptool C routines
try:
    from exptool.basis._accumulate_c import r_to_xi,xi_to_r,d_xi_to_r
except:
    from exptool.basis.compatibility import r_to_xi,xi_to_r,d_xi_to_r





'''
###########################################################################
# methods

parse_slgrid(file)
     take spherical cache and get basic quantities

read_cached_table(file)
     take spherical cache, get basic quantities and tables (returned a matrices)

read_sph_model_table(file)
     take 1d model table and convert using mappings

init_table(modelfile,numr,rmin,rmax)
     generate 1d model table for fitting use


###########################################################################
# examples

import halo_methods

sph_file = '/scratch/mpetersen/Disk001/SLGridSph.cache.run001'
model_file = '/scratch/mpetersen/Disk001/SLGridSph.model'


lmax,nmax,numr,cmap,rmin,rmax,scale = halo_methods.parse_slgrid(sph_file,verbose=0)
lmax,nmax,numr,cmap,rmin,rmax,scale,ltable,evtable,eftable = halo_methods.read_cached_table(sph_file,verbose=0,retall=True)
xi,rarr,p0,d0 = halo_methods.init_table(model_file,numr,rmin,rmax,cmap=cmap,scale=scale)



'''

def read_sph_model_table(modelfile):
    """needs to be made more flexible, but this will at least not break!"""

    A = np.genfromtxt(modelfile,comments='!',skip_header=5)
    print(A[0])
    R1,D1,M1,P1 = A[:,0],A[:,1],A[:,2],A[:,3]
    return R1,D1,M1,P1


def parse_slgrid(file,verbose=0):
    f = open(file,'rb')
    #
    # read the header
    #
    a = np.fromfile(f, dtype=np.uint32,count=4)
    lmax = a[0]
    nmax = a[1]
    numr = a[2]
    cmap = a[3]
    b = np.fromfile(f, dtype=np.float64,count=3)
    rmin = b[0]
    rmax = b[1]
    scale = b[2]

    if verbose > 0:
        print('LMAX=',lmax)
        print('NMAX=',nmax)
        print('NUMR=',numr)
        print('CMAP=',cmap)
        print('RMIN=',rmin)
        print('RMAX=',rmax)
        print('SCALE=',scale)

    f.close()

    return lmax,nmax,numr,cmap,rmin,rmax,scale




def read_cached_table(file,verbose=0,retall=True):
    '''
    read_cached_table
    -----------------------


    inputs
    ----------------------
    file           : string, input file
    verbose        : (integer bit flag, default=0)
    retall         : (boolean, default=True)



    returns
    ----------------------
    if retall:
       lmax
       nmax
       numr
       cmap
       rmin
       rmax
       scale
       ltable
       evtable
       eftable

    if !retall:


    '''
    f = open(file,'rb')
    #
    # read the header
    #
    a = np.fromfile(f, dtype=np.uint32,count=4)
    lmax = a[0]
    nmax = a[1]
    numr = a[2]
    cmap = a[3]
    a = np.fromfile(f, dtype='<f8',count=3) # this must be doubles
    rmin = a[0]
    rmax = a[1]
    scale = a[2]
    #
    # set up the matrices
    #
    ltable = np.zeros(lmax+1)
    evtable = np.ones([lmax+1,nmax])
    eftable = np.ones([lmax+1,nmax,numr])
    #
    #

    for l in range(0,lmax+1):
        #
        # The l integer
        #
        ltable[l] = np.fromfile(f, dtype=np.uint32,count=1)
        evtable[l,0:nmax] = np.fromfile(f,dtype='f8',count=nmax)

        for n in range(0,nmax):
            if verbose==1: print(l,n)
            #
            # loops for different levels go here
            #
            eftable[l,n,:] = np.fromfile(f,dtype='f8',count=numr)

    f.close()

    if retall:
        return lmax,nmax,numr,cmap,rmin,rmax,scale,ltable,evtable,eftable
    else:
        return ltable,evtable,eftable








def init_table(modelfile,numr,rmin,rmax,cmap=0,scale=1.0,spline=True):
    """
    make the model table and the cachefile agree on abscissa
    """
    R1,D1,M1,P1 = read_sph_model_table(modelfile)

    fac0 = 4.*np.pi
    xi = np.zeros(numr)
    r = np.zeros(numr)
    p0 = np.zeros(numr)
    d0 = np.zeros(numr)

    if (cmap==1):
        xmin = (rmin/scale - 1.0)/(rmin/scale + 1.0);
        xmax = (rmax/scale - 1.0)/(rmax/scale + 1.0);

    if (cmap==2):
        xmin = log(rmin);
        xmax = log(rmax);

    if (cmap==0):
        xmin = rmin;
        xmax = rmax;

    dxi = (xmax-xmin)/(numr-1);

    #
    #
    if spline==True:
        pfunc = interpolate.splrep(R1, P1, s=0)
        dfunc = interpolate.splrep(R1, fac0*D1, s=0)
    #
    #
    for i in range(0,numr):#(i=0; i<numr; i++):
        xi[i] = xmin + dxi*i;
        r[i] = xi_to_r(xi[i],cmap,scale);
        if spline==False:
            p0[i] = P1[ (abs(r[i]-R1)).argmin() ]; # this is the spherical potential at that radius
            d0[i] = fac0 * D1[ (abs(r[i]-R1)).argmin() ]; # this is the spherical density at that radius (4pi*dens)
        if spline==True:
            p0[i] = interpolate.splev(r[i], pfunc, der=0)
            d0[i] = interpolate.splev(r[i], dfunc, der=0)
    return xi,r,p0,d0
