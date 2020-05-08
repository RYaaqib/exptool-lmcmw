'''
outcoef.py
(broken out of exptool)

tools to read coefficient files and do rudimentary manipulation.

MSP 30 April 2020

'''

import numpy as np
import os

class OutCoef(object):
    """python reader for outcoef files from exp
    
    see specific calls below for the structure of the returned coefs matrix

    inputs
    ----------
    filename : str
        The outcoef filename to load.



    example usage
    ------------
    SL = outcoef.OutCoef(indir+'outcoef.dark.'+runtag)
    plt.plot(SL.T,SL.coefs[:,0,0])

    """

    def __init__(self, filename):

        # determine whether this is a spherical or cylindrical basis file

        self.coeffile = filename
        
        f = open(self.coeffile)
        
        # check if spherical
        [string1] = np.fromfile(f, dtype='a64',count=1)
        
        f.close()
        
        if string1==b'Sphere SL':
            
            self.basis = 'SphereSL'
            
            print('OutCoef: reading SphereSL coefficients . . .')
            self.read_binary_sl_coefficients()
            
        else:
            
            self.basis = 'Cylinder'
            
            print('OutCoef: reading Cylinder coefficients . . .')
            self.read_binary_eof_coefficients()



    def read_binary_eof_coefficients(self):
        '''
        read_binary_eof_coefficients
        definitions to read EXP-generated binary coefficient files (generated by EmpOrth9thd.cc dump_coefs)
        the file is self-describing, so no other items need to be supplied.

        inputs
        ----------------------
        coeffile   : input coefficient file to be parsed

        returns
        ----------------------
        times      : vector, time values for which coefficients are sampled
        coef_array : (rank 4 matrix)
        0: times
        1: cos/0, sin/1 (note all m=0 sine terms are 0)
        2: azimuthal order
        3: radial order

        '''


        f = open(self.coeffile)

        # get the length of the file
        f.seek(0, os.SEEK_END)
        filesize = f.tell()

        # return to beginning
        f.seek(0)

        [time0] = np.fromfile(f, dtype=np.float,count=1)
        [mmax,nmax] = np.fromfile(f, dtype=np.uint32,count=2)

        # hard-coded to match specifications.
        n_outputs = int(filesize/(8*((mmax+1)+mmax)*nmax + 4*2 + 8))

        # set up arrays given derived quantities
        times = np.zeros(n_outputs)
        coef_array = np.zeros([n_outputs,2,mmax+1,nmax])

        # return to beginning
        f.seek(0)


        for tt in range(0,n_outputs):

            [time0] = np.fromfile(f, dtype=np.float,count=1)
            [dummym,dummyn] = np.fromfile(f, dtype=np.uint32,count=2)

            times[tt] = time0
        
            for mm in range(0,mmax+1):
            
                coef_array[tt,0,mm,:] = np.fromfile(f, dtype=np.float,count=nmax)
            
                if mm > 0:
                    coef_array[tt,1,mm,:] = np.fromfile(f, dtype=np.float,count=nmax)

            
        #return times,coef_array
        self.T = times
        self.coefs = coef_array


    def read_binary_sl_coefficients(self):
        '''
        read_binary_sl_coefficients
        definitions to read EXP-generated binary coefficient files (generated by SphericalBasis.cc dump_coefs)
        the file is self-describing, so no other items need to be supplied.

        inputs
        ----------------------
        coeffile   : input coefficient file to be parsed

        returns
        ----------------------
        T      : vector, time values for which coefficients are sampled
        coefs  : (rank 3 matrix)
           0: times
           1: azimuthal (L) order
           2: radial order

        '''


        f = open(self.coeffile)

        # get the length of the file
        f.seek(0, os.SEEK_END)
        filesize = f.tell()

        # return to beginning
        f.seek(0)

    
        [string1] = np.fromfile(f, dtype='a64',count=1)
        [time0,scale] = np.fromfile(f, dtype=np.float,count=2)
        [nmax,lmax] = np.fromfile(f, dtype=np.uint32,count=2)

        # hard-coded to match specifications.
        n_outputs = int(filesize/(8*(lmax*(lmax+2)+1)*nmax + 4*2 + 8*2 + 64))


        # set up arrays given derived quantities
        times = np.zeros(n_outputs)
        coef_array = np.zeros([n_outputs,lmax*(lmax+2)+1,nmax])


        # return to beginning
        f.seek(0)


        for tt in range(0,n_outputs):
        
            [string1] = np.fromfile(f, dtype='a64',count=1)
            [time0,scale] = np.fromfile(f, dtype=np.float,count=2)
            [nmax,lmax] = np.fromfile(f, dtype=np.uint32,count=2)
        
            times[tt] = time0

            for nn in range(0,nmax):

                coef_array[tt,:,nn] = np.fromfile(f, dtype=np.float,count=lmax*(lmax+2)+1)

        #return times,coef_array
        self.T = times
        self.coefs = coef_array




class CylCoefs(object):
    """python reader for outcoef files from exp
    
    see specific calls below for the structure of the returned coefs matrix

    inputs
    ----------
    filename : str
        The outcoef filename to load.

    example usage
    ------------
    SL = outcoef.Cylcoefs(indir+'outcoef.dark.'+runtag)
    plt.plot(SL.T,SL.coefs[:,0,0])

    """

    def __init__(self, filename):

        self.CylCoefs = OutCoef(filename)

        self.numt   = self.CylCoefs.T
        self.mmax   = self.CylCoefs.coefs.shape[2]
        self.norder = self.CylCoefs.coefs.shape[3]


    def normalise(self):
        """
        apply the monopole normalisation, from the original model
        """

        monopole_norm = self.CylCoefs.coefs[:,0,0,0]

        self.CylNormCoefs = np.zeros([self.numt,2,self.mmax+1,self.norder])

        for i in range(0,self.mmax+1):
            for j in range(0,self.norder):
                self.CylNormCoefs[:,0,i,j] = self.CylCoefs.coefs[:,0,i,j]/monopole_norm
                self.CylNormCoefs[:,1,i,j] = self.CylCoefs.coefs[:,1,i,j]/monopole_norm


    def total_order_power(self):

        return np.sum(coefs[:,0,:,:]*coefs[:,0,:,:] + 
                      coefs[:,1,:,:]*coefs[:,1,:,:],axis=2)

    def total_order_amplitude(self):

        return np.sqrt(np.sum(coefs[:,0,:,:]*coefs[:,0,:,:] + 
                      coefs[:,1,:,:]*coefs[:,1,:,:],axis=2))

    
