import sys
sys.path.append('/home/tnguser/python/')
import numpy as np
import h5py
import os
from FigureRotation_util import *
from Cannon_get_principal_axes import *
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from scipy.optimize import minimize, curve_fit


#from run_steady_rotation import *

def get_basis(vec):
    v1 = np.cross(vec,np.random.normal(0,1,3))
    v1 = v1 / mag(v1)
    
    v2 = np.cross(vec,v1)
    v2 = v2 / mag(v2)
    
    return np.array([v1,v2,vec]).T

def mag(vec):
    
    return np.sqrt(np.sum(vec[:3]**2))

def mag2(vec):
    
    return np.sqrt(np.sum(vec[3:6]**2))

def orthonormalize(E_arr):
    e1_arr = E_arr[:,:,1] - np.diagonal(np.dot(E_arr[:,:,0],E_arr[:,:,1].T)).reshape(-1,1)*E_arr[:,:,0]
    e1_arr = e1_arr / np.sqrt(np.sum(e1_arr**2,axis=1)).reshape(-1,1)

    e2_arr = np.cross(E_arr[:,:,0],e1_arr)
    e2_arr

    E_arr[:,:,1] = e1_arr
    E_arr[:,:,2] = e2_arr
    
    return E_arr


def get_phi(ei_in_ref):
    """
    assume input has two axes
    """
    ref_r = np.copy(ei_in_ref)
    if len(ei_in_ref.shape)==3:
        ref_r[:,:,2] = 0
        
        ref_r = ref_r / np.sqrt(np.sum(ref_r**2,axis=2)).reshape(ref_r.shape[0],ref_r.shape[1],1)
        phi_arr = []
        for j in range(2):
            phi_arr_i = [0]
            for i in range(1,ref_r.shape[1]):
                phi_arr_i.append(np.arcsin(np.cross(ref_r[j,i-1,:],ref_r[j,i,:]))[2])
            phi_arr.append(phi_arr_i)
        return np.cumsum(phi_arr,axis=1)
    else:
        ref_r[:,2] = 0
        ref_r = ref_r / np.sqrt(np.sum(ref_r**2,axis=1)).reshape(-1,1)

        phi_arr = [0]
        for i in range(1,len(ref_r)):
            phi_arr.append(np.arcsin(np.cross(ref_r[i-1,:],ref_r[i,:]))[2])    
        
        return np.cumsum(phi_arr)


def nanarccos(x):
    ind_nan = np.where(np.abs(x)>1)[0]
    x[ind_nan] = x[ind_nan]/np.abs(x[ind_nan])
    return np.arccos(x)

def e0_err(args,c,d):
    p_e,N = args
    return c/np.sinh(d*(1+p_e)) / np.sqrt(N)

def e1_err(args,c,d1,d2):
    p_e,N = args
    return c*(1/np.sinh(d1*(1+p_e))+1/np.sinh(d2*(1-p_e))) / np.sqrt(N)

def e2_err(args,c,d):
    p_e,N = args
    return c/np.sinh(d*(1-p_e)) / np.sqrt(N)

def analytic_error(args,const1,const2,const3):
    
   
    return  np.concatenate((e0_err(args,*const1).reshape(-1,1),
                            e1_err(args,*const2).reshape(-1,1),
                            e2_err(args,*const3).reshape(-1,1)),axis=1)


def y(x,m,b):
    return m*x + b

def chi2_line(y_obs,x,err,params):
    y_pred = y(x,*params)
    return 1/len(y_obs)*np.sum(((y_pred-y_obs)/err)**2)

def sphere2cart(phi,theta):
    """
    theta polar angle, phi azimuthal
    """
    return np.array([np.cos(phi)*np.sin(theta),
                     np.sin(phi)*np.sin(theta),
                     np.cos(theta)])

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

def chi2(bodyAxes,pts,err):
    
    if bodyAxes.ndim > 1:
        X2 = []
        for bodyAxis in bodyAxes:
            #bodyAxes = np.transpose(pts,axes=(0,2,1)).dot(axis)
            
            #rotate the axis out of the body frames
            axis = pts.dot(bodyAxis)
            
            # find the mean of these, they should cluster on the mean
            meanAxis = np.mean(axis,axis=0)
            # NORMALIZE the mean so that it is a unit vector
            meanAxis = meanAxis / np.sqrt(np.sum(meanAxis**2)) 
            
            dev_from_mean = np.arccos(axis.dot(meanAxis))
            
            # Get the expected error from the analytic fits
            # interpolate between the errors on the three principal axes to get the error on the 
            # rotation axis 
            sigma = err.dot(bodyAxis) #1e-2
            
            nu = len(dev_from_mean) - 2 # -2 b/c we have fixed two parameters, phi and theta, of the rotation axis
            
            if nu == 0:
                X2.append(np.inf)
            else:
                X2.append(1/nu*np.sum(dev_from_mean**2/sigma**2)) 
    else:
        #bodyAxes = np.transpose(pts,axes=(0,2,1)).dot(axes)
        axes = pts.dot(bodyAxes)
        meanAxis = np.mean(axes,axis=0)
        meanAxis = meanAxis / np.sqrt(np.sum(meanAxis**2))
        dev_from_mean = np.arccos(axes.dot(meanAxis)) 
        
        sigma = err.dot(bodyAxes) #1e-2
        nu = len(dev_from_mean) - 2
        
        if nu == 0:
            X2 = np.inf
        else:
            X2 = 1/nu*np.sum(dev_from_mean**2/sigma**2)

    return np.array(X2)
            
def squareDev_angular(args,pts):
    
    bodyAxis = sphere2cart(*args)
    
    axes = pts.dot(bodyAxis)
    meanAxis = np.mean(axes,axis=0)
    meanAxis = meanAxis / np.sqrt(np.sum(meanAxis**2))
    dev_from_mean = np.arccos(axes.dot(meanAxis)) 


    X2 = np.sum(dev_from_mean**2)

    return np.array(X2)


####################################################################
if __name__ == '__main__':
    
    overwrite = True

    outerVirialLim = float(sys.argv[2])
    innerVirialLim = float(sys.argv[3])

    #res = sys.argv[1]
    sim = sys.argv[1] #'L35n'+res+'TNG_DM'

    halo_catalog = np.load('/home/tnguser/postprocessing/halo_catalogues/'+sim+'.npy')
    startSnap = 75

    loadDir = "/home/tnguser/postprocessing/principal_axes/"+sim+"/%02d_%02dRvir/"%(innerVirialLim*10,outerVirialLim*10)

    #figDir = "/Users/nfash/valluri/figures/simultaneous_plane_fitting/baryonic_halos/"
    saveDir = "/home/tnguser/postprocessing/pattern_speeds/"+sim+"/%02d_%02dRvir/"%(innerVirialLim*10,outerVirialLim*10)

    snapArr,zArr = snapshot_redshift_corr('/home/tnguser/sims.TNG/' + sim + '/output')

    z0 = zArr[startSnap==snapArr]

    t_arr = []
    for snap in range(startSnap,100):

        z1 = zArr[snap==snapArr]    
        t_arr.append(cosmo.lookback_time(z0) - cosmo.lookback_time(z1))

    t_arr = np.array(t_arr)


    # Get the analytic error fit constants
    errFitConsts = np.load("/home/tnguser/postprocessing/"+\
                               "empirical_error_fits_full.npy",allow_pickle=True)

    # This is a guess axis, for later fitting
    r0 = np.array([1,1,1])
    r0 = r0 / np.sqrt(np.sum(r0**2))

    completeCount = 0
    failedFitList = []
    for GrNr in halo_catalog:

        if os.path.exists(saveDir+'GrNr_%d_snap_%d_99_patternSpeeds.npy'%(GrNr,startSnap)) and not overwrite:
            print("File exists for GrNr %d, skipping"%GrNr)
            completeCount += 1
            continue

        print("beginning work on halo %d\t %d/%d complete"%(GrNr,completeCount,len(halo_catalog)))
        completeCount += 1

        if not os.path.exists(loadDir+"GrNr_%d_snap_%d_99_principal_axes_full.npy"%(GrNr,startSnap)):
            print("No existing eigvecs for halo, skipping")
            completeCount += 1
            continue

        if GrNr == -1:
            print('No match to DM only or selection criteria not met, skipping')
            completeCount += 1
            continue

        axisRatios = np.load(loadDir+"GrNr_%d_snap_%d_99_axisRatios_full.npy"%(GrNr,startSnap),allow_pickle=True)
        p_e = (1-2*axisRatios[:,0] + axisRatios[:,1])/(1-axisRatios[:,1])
        #Npart = np.load(loadDir+"GrNr_%d_snap_%d_99_Npart_leq_06_Rvirial.npy"%(GrNr,startSnap))
        Npart = np.load(loadDir+"GrNr_%d_snap_%d_99_Npart.npy"%(GrNr,startSnap))

        if max(abs(p_e))>= 0.9:
            print("Halo is nearly oblate/prolate, cannot accurately measure")
            continue

        sigma = analytic_error([p_e,Npart],*errFitConsts)

        mean_axes = np.load(loadDir+"GrNr_%d_snap_%d_99_principal_axes_full.npy"%(GrNr,startSnap),allow_pickle=True)

        ###############################################################################################
        #### Begin iterative plane fitting                                                         ####
        ###############################################################################################

        print("Beginning simultaneous plane fitting")
        IC = []
        snapPairs = [[i,i+1] for i in range(len(mean_axes)-1)]

        snapPairs_gen = [np.copy(snapPairs)]
        nIter = 0
        fit_iter_gen = []
        chi2_iter_gen = []
        while nIter < len(mean_axes):

            chi2_iter = []
            chi2_pred = []

            fit_iter = []
            for i in range(len(snapPairs)):
                pair = snapPairs[i]
                # generate the 'fits' between snapshot pairs
                # these are ~ identical to the quaternions

                fit = minimize(squareDev_angular,cart2spherical(r0),args=mean_axes[pair[0]:(pair[1]+1)]) #,
                               #bounds=[(-np.pi,np.pi),(0,np.pi)]) # disable bounds, seems like they cause more problems than help
                ri = sphere2cart(*fit.x)
                
                chi2_iter.append(chi2(ri,mean_axes[pair[0]:(pair[1]+1)],sigma[pair[0]:(pair[1]+1)]))

                fit_iter.append(ri)

                # Check how well we can fit the next set of snapshots if we merge
                if i+1 == len(snapPairs):
                    break
                pairNext = snapPairs[i+1]
                fit = minimize(squareDev_angular,cart2spherical(r0),args=mean_axes[pair[0]:(pairNext[1]+1)])#,
                               #bounds=[(-np.pi,np.pi),(0,np.pi)])
                ri_pred = sphere2cart(*fit.x)


                chi2_pred.append(chi2(ri_pred,mean_axes[(pair[0]):(pairNext[1]+1)],sigma[pair[0]:(pairNext[1]+1)]))

            # Evaluate the BIC
            chi2_iter_gen.append(chi2_iter)

            chi2_net = np.sum(chi2_iter) / len(chi2_iter)
            nSnap = len(snapPairs)

            deltaChi2 = 1
            IC.append([nSnap, nSnap*deltaChi2 + chi2_net])

            fit_iter_gen.append(fit_iter)

            if nSnap == 1:
                break

            # merge where the chi2 is minimum
            ind_min = np.argmin(chi2_pred)

            # update the pairing list
            snapPairs[ind_min] = [snapPairs[ind_min][0],snapPairs[ind_min+1][-1]]
            del(snapPairs[ind_min+1])

            snapPairs_gen.append(np.copy(snapPairs))

            nIter += 1

        IC = np.array(IC)
        minIC = np.argmin(IC[:,1])

        fit_iter = fit_iter_gen[minIC]
        chi2_iter = chi2_iter_gen[minIC]

        print("Finished, found %d distinct rotation axes\nTotal chi^2: %.1f"%(len(snapPairs_gen[minIC]),np.sum(chi2_iter)/len(chi2_iter)))

        ###############################################################################################
        #### Done                                                                                  ####
        ###############################################################################################

        # Place these fit axes into the body frame
        print("Searching for rotation epochs in planes, recovering pattern speeds")

        sigma_deg = sigma * 180/np.pi
        Omega = {'GrNr':GrNr,'Omega':[],'sigma_Omega':[],'chi2':[],'startStop':[],'Raxis':[],
                 'plane_chi2':[]}

        for axis, pair in zip(fit_iter,snapPairs_gen[minIC]):

            if pair[1] == pair[0]+1:
                continue

            # to get the fit rotation axis in the body frame, just rotate it using all of the 
            # body frames and take the mean. This isn't ideal, but it's not a bad approximation either

            axes_box_frame = mean_axes[pair[0]:(pair[1]+1)].dot(axis)

            mean_box_frame_axis = np.mean(axes_box_frame,axis=0)
            mean_box_frame_axis = mean_box_frame_axis / np.sqrt(np.sum(mean_box_frame_axis**2))
            rotBasis = get_basis(mean_box_frame_axis)

            phi = []
            t = []
            sigma_tot = []
            for j in range(3):

                ej = mean_axes[pair[0]:(pair[1]+1),:,j]
                ej_rot = rotBasis.T.dot(ej.T).T
                phi_j = get_phi(ej_rot)*180/np.pi

                phi_j = phi_j - np.mean(phi_j)

                phi.append(phi_j)
                t.append(t_arr[pair[0]:(pair[1]+1)])
                sigma_tot.append(sigma_deg[pair[0]:(pair[1]+1),j])

            phi = np.array(phi)
            t = np.array(t)
            sigma_tot = np.array(sigma_tot)

            # Iterate through the axes, trim which one contributes most to the chi2
            IC_j = []
            X2_j = []
            params_j = []
            covMat_j = []
            indices_j = []
            indices = np.arange(3)
            indices_j = [np.copy(indices)]

            deltaChi2 = 10

            for j in range(3):

                params,covMat = curve_fit(y,t[indices].flatten(),phi[indices].flatten(),
                                          p0=[150/4,0],sigma=sigma_tot[indices].flatten())
                X2_line = chi2_line(phi[indices].flatten(),t[indices].flatten(),sigma_tot[indices].flatten(),params)

                IC_j.append(-len(indices)*deltaChi2 + X2_line)
                X2_j.append(X2_line)
                params_j.append(params)
                covMat_j.append(covMat)

                max_contributor = np.argmax([chi2_line(phi[i],t[i],sigma_tot[i],params) for i in indices])

                indices = np.delete(indices,max_contributor)
                if len(indices)>0:
                    indices_j.append(np.copy(indices))

            IC_argmin = np.argmin(IC_j)
            indices, params = indices_j[IC_argmin], params_j[IC_argmin]
            covMat = covMat_j[IC_argmin]
            X2 = X2_j[IC_argmin]


            #############################################################################################

            Omega['Omega'].append(params[0]); Omega['sigma_Omega'].append(np.sqrt(covMat[0][0])); 
            Omega['chi2'].append(X2_line); 
            Omega['startStop'].append(pair)
            Omega['Raxis'].append(axis);
        Omega['plane_chi2'].append(chi2_iter)

        np.save(saveDir+'GrNr_%d_snap_%d_99_patternSpeeds.npy'%(GrNr,startSnap),Omega,allow_pickle=True)
        print("Output saved, finished")
