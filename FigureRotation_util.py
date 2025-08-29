#!/usr/bin/env python

# Various functions and tools for measuring halo figure rotation

import numpy as np

def rotationMatrix(alpha,beta,gamma):
    """
    creates rotation matrix for alpha about z axis, beta about y axis, gamma about x axis
    """
    sinalpha,sinbeta,singamma = np.sin([alpha, beta, gamma])
    cosalpha,cosbeta,cosgamma = np.cos([alpha, beta, gamma])

    return np.array([[cosalpha*cosbeta, cosalpha*sinbeta*singamma-sinalpha*cosgamma, cosalpha*sinbeta*cosgamma+sinalpha*singamma],
                     [sinalpha*cosbeta, sinalpha*sinbeta*singamma+cosalpha*cosgamma, sinalpha*sinbeta*cosgamma-cosalpha*singamma],
                     [-sinbeta        , cosbeta*singamma                           , cosbeta*cosgamma]])


def make_halo(nbody=1e5,axis_ratios=np.array([1.0, 0.7, 0.4]),rotate=True):
    # I lifted this from measureshape.py
    # create a triaxial and rotated Hernquist model
    nbody    = int(nbody)
    radius   = 1 / (np.random.random(size=nbody)**-0.5 - 1)
    costheta = np.random.uniform(-1, 1, size=nbody)
    sintheta = (1-costheta**2)**0.5
    phi      = np.random.uniform(0, 2*np.pi, size=nbody)
    pos      = np.column_stack((
      radius * sintheta * np.cos(phi),
      radius * sintheta * np.sin(phi),
      radius * costheta ))
    pos     *= axis_ratios # make it triaxial
    if rotate:
        rot_mat  = rotationMatrix(*list(np.random.random(size=3)))
        pos      = rot_mat.dot(pos.T).T  # rotate by some random angles
    mass     = np.ones(nbody) / nbody

    if rotate:
        return pos, mass, rot_mat
    else:
        return pos, mass


def get_evec(pos, mass, radius,evecGuess = np.eye(3),axesGuess = np.ones(3),r_inner=None, r_norm=False, shell=True):
    # From measureshape.py, but instead returns the eigenvalues and vectors from
    # the measured halo shape

    evec = evecGuess   # initial guess for axes orientation
    axes = axesGuess  # and axis ratios; these are updated at each iteration
    while True:
        # use particles within the elliptical radius less than the provided value
        ellpos  = pos[:,0:3].dot(evec) / axes

        if not shell:
            filter  = np.sum(ellpos**2, axis=1) < radius**2
        elif r_inner is None:
            raise Exception("Inner shell radius must be specified with shell = True")
        else:
            filter = (np.sum(ellpos**2, axis=1) < radius**2) & (np.sum(ellpos**2, axis=1) >= r_inner**2)

        if r_norm:
            r2      = np.sum(pos[filter,0:3]**2,axis=1).reshape(-1,1)
            inertia = pos[filter,0:3].T.dot(pos[filter,0:3] * mass[filter,None]/r2)
        else:
            inertia = pos[filter,0:3].T.dot(pos[filter,0:3] * mass[filter,None])

        val,vec = np.linalg.eigh(inertia)
        order   = np.argsort(-val)  # sort axes in decreasing order
        evec    = vec[:,order]         # updated axis directions
        axesnew = (val[order] / np.prod(val)**(1./3))**0.5  # updated axis ratios, normalized so that ax*ay*az=1
        #print evec,axesnew,sum(filter)
        if sum(abs(axesnew-axes))<0.01: break
        axes    = axesnew

    return val[order],evec #val[sort_inds],vec[:,sort_inds]

def recoverAngles(rot_mat,deg=False):
    """
    Recover the angles rotated about each axis for a given rotation defined by rot_mat
    """
    beta  = np.arcsin(-rot_mat[2,0])
    gamma = np.arctan(rot_mat[2,1]/rot_mat[2,2])
    alpha = np.arctan(rot_mat[1,0]/rot_mat[0,0])

    return np.array([alpha,beta,gamma]) *180/np.pi if deg else np.array([alpha,beta,gamma])

def recoverAngles_agama(rot_mat):
    """
    Recover the angles rotated about axes x,y,z (in order)

    This isn't working at the moment, don't use
    """
    alpha = np.arctan(-rot_mat[2,0]/rot_mat[2,1])
    beta  = np.arccos(rot_mat[2,2])
    gamma = np.arctan(rot_mat[0,2]/rot_mat[1,2])

    return np.array([alpha,beta,gamma])

def orient(E,E_ref=np.eye(3),assert_on=True):
    """
    Enforces E to be a RHS, then attempts rotations by 180 deg about each axis until
    the angles with respect to E_ref are minimized. If any of these cannot be reduced beyond 90 deg,
    throw error
    """

    # Enforce RHS
    RHS = np.dot(np.cross(E[:,0],E[:,1]),E[:,2]) > 0
    if not RHS: E[:,0] = E[:,0]*-1

    # Begin iteration
    # Check base orientation first
    rotations = [[ 1, 1, 1],
                 [-1,-1, 1],
                 [ 1,-1,-1],
                 [-1, 1,-1]]

    angles = []
    for rot in rotations:
        E_rot = E * rot

        # Check angles between E, E_ref
        angles.append(np.arccos(np.diagonal(E_rot.T.dot(E_ref))))

    # Find the rotation which gives us the smallest square error from E_ref
    angles = np.array(angles)
    mean_sqr_dev = np.mean(angles**2,axis=1)

    min_ind = np.argmin(mean_sqr_dev)
    E_new = E*rotations[min_ind]

    # Verify none of the angles on this surpass 90 deg
    #print(angles[min_ind]*180/np.pi)
    if assert_on:
        assert np.all(angles[min_ind]<np.pi/2)

    return E_new

def quaternion(R,alpha_pos=True):
    """
    Returns the quaternion representation of the rotation matrix R

    Does this by first identifying the (only) real eigenvector, which is
    necessarily the rotation axis (since it is the only invariant vector under
    the rotation)

    """

    Eval,Evec = np.linalg.eig(R)

    real_ind = [all(np.isreal(Evec)[:,i]) for i in range(3)]
    assert not all(real_ind)

    rot_axis = np.real(Evec[:,real_ind]).reshape(3)

    # Next, find the angle of rotation about this
    # First pick out a vector within the rotation plane

    n1 = np.cross(rot_axis,np.random.random(3))
    n1 = n1 / np.sqrt(np.sum(n1**2))
    n2 = R.dot(n1)

    # Find alpha and its direction

    cos_alpha = np.dot(n1.reshape(3),n2.reshape(3))
    sin_alpha = np.dot(rot_axis,np.cross(n1.reshape(3),n2.reshape(3)))

    #np.sqrt(np.sum(np.cross(n1.reshape(3),
    #                        n2.reshape(3))**2))

    alpha = np.arctan2(sin_alpha,cos_alpha)

    if alpha_pos and (alpha<0):
        alpha *= -1
        rot_axis = rot_axis * -1

    return rot_axis, alpha

def cart2spherical(arr):
    """
    Convenience function, converts unit vectors in arr to spherical polar
    coordinates on the unit sphere, conserving the quadrant along phi
    """
    if len(arr.shape) < 2:
        arr = arr.reshape(1,-1)

    arr = arr / np.sqrt(np.sum(arr**2,axis=1)).reshape(-1,1)

    theta = np.arccos(arr[:,2])
    phi = np.arctan2(arr[:,1],arr[:,0])

    if len(theta) < 2:
        theta = theta[0]
        phi   = phi[0]

    return phi, theta

def find_rad_to_n(pos,n=.9):
    """
    Returns the radius containing the fraction n of the total population
    """
    dists_sqr = np.sum(pos**2,axis=1)
    sort_inds = np.argsort(dists_sqr)

    ind_to_limit = int(len(pos)*n)
    return np.sqrt(dists_sqr[ind_to_limit])

def rotationMatrix_about_axis(R_axis,omega=1):
    """
    Generates a rotation matrix which will perform a rotation about R_axis by
    an amount omega
    """
    # force R_axis to be unit length
    R_axis = R_axis / np.sqrt(np.sum(R_axis**2))

    # convert omega to rad
    omega = omega *np.pi/180

    # Construct basis on R_axis
    # first cross R_axis with some random vector, to find a normal vector
    R2_axis = np.cross(np.random.random(3),R_axis)
    R2_axis = R2_axis / np.sqrt(np.sum(R2_axis**2))

    R3_axis = np.cross(R_axis,R2_axis)
    R3_axis = R3_axis / np.sqrt(np.sum(R3_axis**2))

    R_basis = np.array([R_axis,R2_axis,R3_axis]).T

    assert np.dot(np.cross(R_basis[:,0],R_basis[:,1]),R_basis[:,2]) > 0

    rot_in_R = rotationMatrix(0,0,omega)

    # Transform the rotation matrix to the rest frame
    return R_basis.dot(rot_in_R).dot(np.linalg.inv(R_basis))

def rotate_halo(pos,rot=None):
    # Just rotate it a little
    # by some random amount
    # we'll save the rotation though
    if rot is None:
        axis = np.random.normal(loc=0,scale=1,size=3)
        axis = axis / np.sqrt(np.sum(axis**2))

        axis2 = np.cross(np.random.normal(loc=0,scale=1,size=3),axis)
        axis2 = axis2 / np.sqrt(np.sum(axis2**2))

        axis3 = np.cross(axis,axis2)
        axis3 = axis3 / np.sqrt(np.sum(axis3**2))

        # This is the transpose of the basis matrix
        rot_mat = np.array([axis,axis2,axis3]).T

    else:
        rot_mat = rotationMatrix(rot[0],rot[1],rot[2])

    return rot_mat.dot(pos.T).T, rot_mat

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    #plt.style.use('dark_background')
    import h5py

    snap = 99
    subhalo_ID = 10
    snapPath = "/Users/nfash/Documents/Valluri/valluri/cutouts/TNG50-1-Dark/Snapshot%d/cutout_%d.hdf5"%(snap,subhalo_ID)

    E_arr = [np.eye(3)]

    snapshot = h5py.File(snapPath,'r')
    pos = snapshot['PartType1']['Coordinates'][:]
    mass = np.ones(len(pos))

    # Subtract of the mean to center the halo
    center = np.mean(pos,axis=0)
    pos = pos - center


    #################################################################################
    ####         Get principal axes                                              ####
    #################################################################################
    temp_E_arr = []
    temp_axis_ratio_arr = []

    red_E_arr = []
    red_axis_ratio_arr = []

    r_outer = np.max(np.sqrt(np.sum(pos**2,axis=1)))
    r_shells = np.logspace(0,2,12)

    for i in range(len(r_shells)-1):

        eigval,E = get_evec(pos,mass,r_shells[i+1],
                            r_inner=r_shells[i],shell=False,r_norm=False)

        E = orient(E,E_arr[-1],assert_on=False)

        red_eigval,red_E = get_evec(pos,mass,r_shells[i+1],
                            r_inner=r_shells[i],shell=False,r_norm=True)

        E = orient(red_E,E_arr[-1],assert_on=False)

        temp_E_arr.append(E)
        temp_axis_ratio_arr.append(np.sqrt(eigval))

        red_E_arr.append(red_E)
        red_axis_ratio_arr.append(red_eigval)


    x0 = temp_E_arr[0][:,0]
    xi = np.array([temp_E_arr[i][:,0] for i in range(len(temp_E_arr))])
    #x_offset = np.arccos(xi.dot(x0)) * 180/np.pi
    x_cosTheta = xi.dot(x0)

    y0 = temp_E_arr[0][:,1]
    yi = np.array([temp_E_arr[i][:,1] for i in range(len(temp_E_arr))])
    #y_offset = np.arccos(yi.dot(y0)) * 180/np.pi
    y_cosTheta = yi.dot(y0)

    z0 = temp_E_arr[0][:,2]
    zi = np.array([temp_E_arr[i][:,2] for i in range(len(temp_E_arr))])
    #z_offset = np.arccos(zi.dot(z0)) * 180/np.pi
    z_cosTheta = zi.dot(z0)

    x0_red = red_E_arr[0][:,0]
    xi_red = np.array([red_E_arr[i][:,0] for i in range(len(red_E_arr))])
    #x_offset = np.arccos(xi.dot(x0)) * 180/np.pi
    x_cosTheta_red = xi_red.dot(x0_red)

    y0_red = red_E_arr[0][:,1]
    yi_red = np.array([red_E_arr[i][:,1] for i in range(len(red_E_arr))])
    #x_offset = np.arccos(xi.dot(x0)) * 180/np.pi
    y_cosTheta_red = yi_red.dot(y0_red)

    z0_red = red_E_arr[0][:,2]
    zi_red = np.array([red_E_arr[i][:,2] for i in range(len(red_E_arr))])
    #x_offset = np.arccos(xi.dot(x0)) * 180/np.pi
    z_cosTheta_red = zi_red.dot(z0_red)

    fig = plt.figure(figsize=(9,10))
    axes = fig.subplots(2,1)

    axes[0].plot(r_shells[1:],x_cosTheta,label='Major',color='gray')
    axes[0].plot(r_shells[1:],y_cosTheta,label="Intermediate")
    axes[0].plot(r_shells[1:],z_cosTheta,label='Minor')

    axes[0].plot(r_shells[1:],x_cosTheta_red,label='Major (red)',color='gray',linestyle='dotted')
    axes[0].plot(r_shells[1:],y_cosTheta_red,label="Intermediate (red)",color='C0',linestyle='dotted')
    axes[0].plot(r_shells[1:],z_cosTheta_red,label='Minor (red)',color='C1',linestyle='dotted')

    axes[0].set_xlabel(r"Outer shell radius")
    axes[0].set_ylabel(r"Cosine axis offset from inner halo")
    axes[0].legend();

    temp_axis_ratio_arr = np.array(temp_axis_ratio_arr) / np.array(temp_axis_ratio_arr)[:,0].reshape(-1,1)
    red_axis_ratio_arr = np.array(red_axis_ratio_arr) / np.array(red_axis_ratio_arr)[:,0].reshape(-1,1)

    axes[1].plot(r_shells[1:],temp_axis_ratio_arr[:,1],label="Intermediate")
    axes[1].plot(r_shells[1:],temp_axis_ratio_arr[:,2],label='Minor')

    axes[1].plot(r_shells[1:],red_axis_ratio_arr[:,1],label="Intermediate (red)",ls='dotted',color='C0')
    axes[1].plot(r_shells[1:],red_axis_ratio_arr[:,2],label='Minor (red)',ls='dotted',color='C1')

    axes[1].set_xlabel(r"Outer shell radius")
    axes[1].set_ylabel("Axis ratios")

    axes[0].set_xscale('log')
    axes[1].set_xscale('log')
