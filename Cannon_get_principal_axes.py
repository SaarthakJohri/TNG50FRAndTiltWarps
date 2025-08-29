import sys
sys.path.append('/home/tnguser/python/')
import illustris_python as il
import numpy as np

import h5py
import os
#from subhalo_merger_tree import *
from FigureRotation_util import *
#from Cannon_build_halo_catalogue import *
from astropy import constants as const
from astropy import units as u



def snapshot_redshift_corr(basePath,startSnap=75):
    redshift_space = []
    for snapshot_number in range(startSnap,100):
        header=il.groupcat.loadHeader(basePath,snapshot_number)
        redshift_space.append(header.get('Redshift'))
    return np.arange(startSnap,100), np.array(redshift_space)

def value(arr):
    """
    Convenience function, returns the item contained by an array,
    or an error if the array has more than one element
    """
    assert len(arr)<2
    return arr[0]

#######################################################################################
if __name__ == '__main__':
    res = sys.argv[1]
    sim = 'L35n'+res+'TNG_DM'
    basePath = '/home/tnguser/sims.TNG/' + sim + '/output'

    catalogue_path = '/home/tnguser/postprocessing/halo_catalogues/' + sim + '.npy'
    naive_halos = np.load(catalogue_path)


    save_name = '/home/tnguser/postprocessing/principal_axes/' + sim + '/%02d_%02dRvir/'
    if not os.path.exists(save_name):
        os.mkdir(save_name)

    working_dir = os.getcwd()

    startSnap = 75
    rnorm = False

    restart = True
    if restart:
        if os.path.exists('halo_err_est_restart.npy'):
            completed_halos = np.load('halo_err_est_restart.npy')
        else:
            completed_halos = []


    print('getting redshift/snapshot relation, particle mass...')

    h0 = 0.6774
    snapArr,zArr = snapshot_redshift_corr(basePath)
    m = 1 #get('http://www.tng-project.org/api/'+sim+'/')['mass_dm'] *1e10/h0
    DM = 1
    print('done')

    print('Accessing group catalogues')
    subhaloGrNr = {}
    subhaloLen = {}
    HaloRvir = {}
    subhaloPos = {}
    for snap in range(startSnap,100):
        subhaloGrNr[snap] = il.groupcat.loadSubhalos(basePath,snap,'SubhaloGrNr')
        subhaloLen[snap] = il.groupcat.loadSubhalos(basePath,snap,'SubhaloLen')
        HaloRvir[snap] = il.groupcat.loadHalos(basePath,snap,'Group_R_Crit200')
        subhaloPos[snap] = il.groupcat.loadSubhalos(basePath,snap,'SubhaloPos')

    groupFirstSub = il.groupcat.loadHalos(basePath,99,'GroupFirstSub')
    print('done')

    finishedCount = 0
    for GrNr in naive_halos:

        if restart and (GrNr in completed_halos):
            finishedCount += 1
            continue

        print("Beginning work on halo %d (%d/%d complete)"%(GrNr,finishedCount,len(naive_halos)))
        finishedCount+= 1

        print("retrieving cutout particles...")

        pos = {}
        #vel = {}
        subfindID = groupFirstSub[GrNr]
        haloTree = il.lhalotree.loadTree(basePath,99,subfindID,fields=['SubhaloGrNr','SnapNum','Group_M_Crit200'], onlyMPB=True)
        haloInd,mpb_snapArr,mass = haloTree['SubhaloGrNr'], haloTree['SnapNum'], haloTree['Group_M_Crit200']

        for snap in range(startSnap,100):

            GrNr_i = value(haloInd[snap == mpb_snapArr])
            pos[snap] = il.snapshot.loadHalo(basePath,snap,GrNr_i,DM,fields='Coordinates')

        print('done')
        print('retrieving subhalo offsets...')

        mainSubLen = {}
        Rvirial = {}
        minPot = {}
        for snap in range(startSnap,100):

            GrNr_i = value(haloInd[snap == mpb_snapArr])
            inGroup = np.where(subhaloGrNr[snap] == GrNr_i)[0]

            lenInGroup = subhaloLen[snap][inGroup]
            mainSubLen[snap] = lenInGroup[0]
            Rvirial[snap] = HaloRvir[snap][GrNr_i]
            particleMinPot = subhaloPos[snap][inGroup[0]]


            a = 1/(1+value(zArr[snap==snapArr]))
            particleCoords = pos[snap]

            ########################################################################
            # check that the halo is not crossing the box boundary, correct if it is
            # condition is whether it is within 2 virial radii
            passUpperBound = particleMinPot + 2*Rvirial[snap] >= 35000
            passLowerBound = particleMinPot - 2*Rvirial[snap] <= 0
            if any(passUpperBound):
                #shift the particles
                particleCoords[:,passUpperBound] = particleCoords[:,passUpperBound] - 35000/2
                particleMinPot[passUpperBound] = particleMinPot[passUpperBound] - 35000/2
                #reenforce boundary condition
                ind = np.where(particleCoords[:,passUpperBound]<0)[0]
                particleCoords[ind,passUpperBound] = particleCoords[ind,passUpperBound] + 35000
            if any(passLowerBound):
                particleCoords[:,passLowerBound] = particleCoords[:,passLowerBound] + 35000/2
                particleMinPot[passLowerBound] = particleMinPot[passLowerBound] + 35000/2

                ind = np.where(particleCoords[:,passLowerBound]>35000)[0]
                particleCoords[ind,passLowerBound] = particleCoords[ind,passLowerBound] - 35000
            # done
            ########################################################################

            particleCoords = (particleCoords[:mainSubLen[snap]] - particleMinPot)*a/h0
            pos[snap] = particleCoords

        print('done')
        print('retreiving eigenvalues/vectors...')

        for [innerVirialLim,outerVirialLim] in [[0,.6],[.3,.6],[.1,.3],[0,.1]]:

            eigvalRatArr = []
            EArr = []
            Npart = []
            for snap in range(startSnap,100):

                a = 1/(1+value(zArr[snap==snapArr]))

                rOuter = outerVirialLim*Rvirial[snap]*a/h0
                rInner = innerVirialLim*Rvirial[snap]*a/h0
                particleCoords = pos[snap]
                mass = np.ones(len(particleCoords))

                r = np.sqrt(np.sum(particleCoords**2,axis=1))
                Npart.append(len(np.where((r<=rOuter)&(r>rInner))[0]))

                # record the full halo  values

                eigval,E = get_evec(particleCoords,mass,rOuter,r_inner=rInner,shell=False,r_norm=rnorm)
                eigvalRatArr.append([eigval[1]/eigval[0],eigval[2]/eigval[0]])
                EArr.append(E)
                ####
                print("finished snap %d"%snap)

            print('done')
            print('orienting frames')
            try:
                i = 0 
                EArrOrient = []
                for E in EArr:

                    if i <1: 
                        E = orient(E,E_ref=E,assert_on=False)
                    else:
                        E = orient(E,E_ref=EArrOrient[-1])

                    EArrOrient.append(E)
                    i+=1

                EArrOrient = np.array(EArrOrient)
            except:
                print('Failed to orient frames on halo %d, skipping'%GrNr)
                if os.path.exists(save_name%(10*innerVirialLim,10*outerVirialLim)+'get_principal_axes_failed_halos.npy'):
                    failed_halos = np.load(save_name%(10*innerVirialLim,10*outerVirialLim)+'get_principal_axes_failed_halos.npy')
                else:
                    failed_halos = []
                failed_halos = np.append(failed_halos,GrNr)
                np.save(save_name%(10*innerVirialLim,10*outerVirialLim)+'get_principal_axes_failed_halos.npy',failed_halos)
                continue

            print('done')
            print('saving results')

            np.save(save_name%(10*innerVirialLim,10*outerVirialLim)+\
                    'GrNr_%d_snap_%d_99_principal_axes_full.npy'%(GrNr,startSnap),EArrOrient,allow_pickle=True)

            # get the axis ratios
            axisRatios = np.sqrt(eigvalRatArr)
            np.save(save_name%(10*innerVirialLim,10*outerVirialLim)+\
                    'GrNr_%d_snap_%d_99_axisRatios_full.npy'%(GrNr,startSnap),axisRatios)
            np.save(save_name%(10*innerVirialLim,10*outerVirialLim)+\
                    "GrNr_%d_snap_%d_99_Npart.npy"%(GrNr,startSnap),Npart)

        if restart:
            completed_halos = np.append(completed_halos,GrNr)
            np.save('get_principal_axes_restart.npy',completed_halos)


          
