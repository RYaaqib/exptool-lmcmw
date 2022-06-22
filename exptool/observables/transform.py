
'''

transform.py : part of exptool

two pieces:
1. to view a galaxy on the sky in some way (LOS usefulness)
2. to convert simulation coordinates into physical units


TODO:
1. add single point rotation
2. add minimum finder for rotation (both trajectory and point)
3. make velocity optional

'''

# standard libraries
import numpy as np
import time

# exptool libraries
from exptool.io import particle


def rotate_point_vector(A,xrotation,yrotation,zrotation,euler=False):
    '''
    rotate_point_vector
        take a collection of 3d points and return the positions rotated by a specified set of angles

    inputs
    ------------------
    A           : input set of points
    xrotation   : rotation into/out of page around x axis, in degrees
    yrotation   : rotation into/out of page around y axis, in degrees
    zrotation   : rotation in the plane of the page (z axis), in degrees
    euler       : boolean
        if True, transform as ZXZ' convention


    returns
    ------------------
    B           : the rotated phase-space output


    '''

    x,y,z = A

    radfac = np.pi/180.

    # set rotation in radians
    a = xrotation*radfac#np.pi/2.2  # xrotation (the tip into/out of page)
    b = yrotation*radfac#np.pi/3.   # yrotation
    c = zrotation*radfac#np.pi      # zrotation

    # construct the rotation matrix TAIT-BRYAN method (x-y-z,
    # extrinsic rotations)
    Rx = np.array([[1.,0.,0.],[0.,np.cos(a),np.sin(a)],[0.,-np.sin(a),np.cos(a)]])
    Ry = np.array([[np.cos(b),0.,-np.sin(b)],[0.,1.,0.],[np.sin(b),0.,np.cos(b)]])
    Rz = np.array([[np.cos(c),np.sin(c),0.,],[-np.sin(c),np.cos(c),0.],[0.,0.,1.]])
    Rmatrix = np.dot(Rx,np.dot(Ry,Rz))

    # construct the rotation matrix EULER ANGLES (z-x-z) (phi, theta,
    # psi)
    # follow the Wolfram Euler angle conventions
    if euler:
        phi = a
        theta = b
        psi = c
        D = np.array([[np.cos(phi),np.sin(phi),0.,],[-np.sin(phi),np.cos(phi),0.],[0.,0.,1.]])
        C = np.array([[1.,0.,0.],[0.,np.cos(theta),np.sin(theta)],[0.,-np.sin(theta),np.cos(theta)]])
        B = np.array([[np.cos(psi),np.sin(psi),0.,],[-np.sin(psi),np.cos(psi),0.],[0.,0.,1.]])
        Rmatrix = np.dot(B,np.dot(C,D))


    # structure the points for rotation
    pts = np.array([x,y,z])

    #
    # do the transformation in position
    tmp = np.dot(pts.T,Rmatrix)
    xout = tmp[:,0]
    yout = tmp[:,1]
    zout = tmp[:,2]
    #

    return [xout,yout,zout]




def rotate_points(PSPDump,xrotation,yrotation,zrotation,velocity=True,euler=False):
    '''
    rotate_points
        take a PSP dump and return the positions/velocities rotated by a specified set of angles

    inputs
    ------------------
    PSPDump     : input set of points
    xrotation   : rotation into/out of page, in degrees
    yrotation   :
    zrotation   :
    velocity    : boolean
        if True, return velocity transformation as well
    euler       : boolean
        if True, transform as ZXZ' convention


    returns
    ------------------
    PSPOut      : the rotated phase-space output


    '''

    radfac = np.pi/180.

    # set rotation in radians
    a = xrotation*radfac#np.pi/2.2  # xrotation (the tip into/out of page)
    b = yrotation*radfac#np.pi/3.   # yrotation
    c = zrotation*radfac#np.pi      # zrotation

    # construct the rotation matrix TAIT-BRYAN method (x-y-z,
    # extrinsic rotations)
    Rx = np.array([[1.,0.,0.],[0.,np.cos(a),np.sin(a)],[0.,-np.sin(a),np.cos(a)]])
    Ry = np.array([[np.cos(b),0.,-np.sin(b)],[0.,1.,0.],[np.sin(b),0.,np.cos(b)]])
    Rz = np.array([[np.cos(c),np.sin(c),0.,],[-np.sin(c),np.cos(c),0.],[0.,0.,1.]])
    Rmatrix = np.dot(Rx,np.dot(Ry,Rz))

    # construct the rotation matrix EULER ANGLES (z-x-z) (phi, theta,
    # psi)
    # follow the Wolfram Euler angle conventions
    if euler:
        phi = a
        theta = b
        psi = c
        D = np.array([[np.cos(phi),np.sin(phi),0.,],[-np.sin(phi),np.cos(phi),0.],[0.,0.,1.]])
        C = np.array([[1.,0.,0.],[0.,np.cos(theta),np.sin(theta)],[0.,-np.sin(theta),np.cos(theta)]])
        B = np.array([[np.cos(psi),np.sin(psi),0.,],[-np.sin(psi),np.cos(psi),0.],[0.,0.,1.]])
        Rmatrix = np.dot(B,np.dot(C,D))


    # structure the points for rotation

    # note: no guard against bad PSP here.
    pts = np.array([PSPDump.xpos,PSPDump.ypos,PSPDump.zpos])

    #
    # instantiate new blank PSP item
    PSPOut = particle.holder()

    #
    # do the transformation in position
    tmp = np.dot(pts.T,Rmatrix)
    PSPOut.xpos = tmp[:,0]
    PSPOut.ypos = tmp[:,1]
    PSPOut.zpos = tmp[:,2]
    #

    # and velocity
    if velocity:
        vpts = np.array([PSPDump.xvel,PSPDump.yvel,PSPDump.zvel])
        tmp = np.dot(vpts.T,Rmatrix)
        PSPOut.xvel = tmp[:,0]
        PSPOut.yvel = tmp[:,1]
        PSPOut.zvel = tmp[:,2]
    #

    return PSPOut



def de_rotate_points(PSPDump,xrotation,yrotation,zrotation,velocity=True):
    '''
    rotate_points
        take a PSP dump and return the positions/velocities derotated by a specified set of angles

    simply the inverse of rotate_points.

    inputs
    ------------------
    PSPDump
    xrotation   : rotation into/out of page, in degrees
    yrotation
    zrotation



    returns
    ------------------
    PSPOut      : the rotated phase-space output


    '''

    radfac = np.pi/180.

    # set rotation in radians
    a = xrotation*radfac#np.pi/2.2  # xrotation (the tip into/out of page)
    b = yrotation*radfac#np.pi/3.   # yrotation
    c = zrotation*radfac#np.pi      # zrotation

    # construct the rotation matrix
    Rx = np.array([[1.,0.,0.],[0.,np.cos(a),np.sin(a)],[0.,-np.sin(a),np.cos(a)]])
    Ry = np.array([[np.cos(b),0.,-np.sin(b)],[0.,1.,0.],[np.sin(b),0.,np.cos(b)]])
    Rz = np.array([[np.cos(c),np.sin(c),0.,],[-np.sin(c),np.cos(c),0.],[0.,0.,1.]])
    Rmatrix = np.linalg.inv(np.dot(Rx,np.dot(Ry,Rz)))

    # structure the points for rotation

    # note: no guard against bad PSP here.
    pts = np.array([PSPDump.xpos,PSPDump.ypos,PSPDump.zpos])

    #
    # instantiate new blank PSP item
    PSPOut = particle.holder()

    #
    # do the transformation in position
    tmp = np.dot(pts.T,Rmatrix)
    PSPOut.xpos = tmp[:,0]
    PSPOut.ypos = tmp[:,1]
    PSPOut.zpos = tmp[:,2]
    #

    # and velocity
    if velocity:

        vpts = np.array([PSPDump.xvel,PSPDump.yvel,PSPDump.zvel])
        tmp = np.dot(vpts.T,Rmatrix)
        PSPOut.xvel = tmp[:,0]
        PSPOut.yvel = tmp[:,1]
        PSPOut.zvel = tmp[:,2]
    #

    PSPOut.mass = PSPDump.mass

    return PSPOut






def compute_spherical(x,y,z,vx,vy,vz,usecartesian=False):
    """return the spherical coordinates and corresponding velocities
    also optionally return the re-computed cartesian for checking.

    inputs
    ----------------
    x
    y
    z
    vx
    vy
    vz
    usecartesian=False

    returns
    ----------------
    r3
    theta
    phi
    vr
    vtheta
    vphi,
    if usecartesian=True:
       xvel
       yvel
       zvel


    """

    r3 = np.sqrt(x*x + y*y + z*z)
    r2 = np.sqrt(z*x + y*y)

    # azimuthal angle
    phi   = np.arctan2(y,x)

    # polar angle
    theta = np.arccos(-z/r3) - np.pi/2.

    cost = (z/(r3+1.e-18))
    sint = np.sqrt(1. - cost*cost)
    cosp = np.cos(phi)
    sinp = np.sin(phi)

    vr      = sint*(cosp*vx + sinp*vy) + cost*vz
    vphi    = (-sinp*vx + cosp*vy)
    vtheta  = (cost*(cosp*vx + sinp*vy) - sint*vz)

    xvel = vr * sint*cosp + vtheta * cost*cosp - sinp*vphi
    yvel = vr * sint*sinp + vtheta * cost*sinp + cosp*vphi
    zvel = vr * cost      - vtheta * sint

    if usecartesian:
        return r3,theta,phi,vr,vtheta,vphi,xvel,yvel,zvel
    else:
        return r3,theta,phi,vr,vtheta,vphi
