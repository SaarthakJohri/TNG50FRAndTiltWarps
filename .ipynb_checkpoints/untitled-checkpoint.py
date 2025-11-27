import sys
import agama
import h5py
import illustris_python as il 
import numpy as np
from scipy.spatial.transform import Rotation as R 
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const

sim = 'L35n2160TNG' # simulation name
basePath = '/home/tnguser/sims.TNG/' + sim + '/output' #calling on simulations
catalogue_path = '/home/tnguser/postprocessing/halocatalogues/' + sim + '.npy'
naive_halos = np.load(catalogue_path)

def snapshot_redshift_corr(basePath,startSnap=75):
    """
    Calls and stores z = and redshift values
    """
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

def make_unit(vec):
    """
    Convenience function, returns the unit-normalized vector
    """
    return 1/np.sqrt(np.sum(vec**2)) * vec

agama.setUnits(length=1, velocity=1, mass=1e10)
main_subfindIDs = [j for i,j in enumerate(main_subhalos) if j!=-1]

final_diskyIDs = []
NaN_fraction = []
total_masses = []
for subfindID in tqdm(main_subfindIDs[:882]):
    GrNr = (il.groupcat.loadSingle(basePath, 99, subhaloID=subfindID)['SubhaloGrNr'])

    snapArr,zArr = snapshot_redshift_corr(basePath)

    haloTree = il.lhalotree.loadTree(basePath,99,subfindID,
                                 fields=['SubhaloGrNr','SnapNum','SubhaloPos','SubhaloNumber','SubhaloHalfmassRadType'],onlyMPB=True)

    snap = 75
    haloInd,mpb_snapArr = haloTree['SubhaloGrNr'], haloTree['SnapNum']
    subfindID = haloTree['SubhaloNumber']
    subhaloPos = haloTree['SubhaloPos']
    halfmassrad = haloTree['SubhaloHalfmassRadType']
    h = 0.678

    GrNr_i       = value(haloInd[snap == mpb_snapArr])
    subfindID_i  = value(subfindID[snap == mpb_snapArr])
    subhaloPos_i = value(subhaloPos[snap == mpb_snapArr])
    a = 1/(1+value(zArr[snap==snapArr]))
    halfmassrad_i = value(halfmassrad[snap == mpb_snapArr])[4]*a

    starPos = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Coordinates')
    starMass = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Masses') 
    starVel = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Velocities')
    starU = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Potential')
    DMPos = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,1,fields='Coordinates')
    gasPos = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,0,fields='Coordinates')
    gasMass = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,0,fields='Masses') 

    particleCoords_bar  = starPos
    particleVels_bar    = starVel #km/s
    particleMass_bar    = starMass * h #1e0 M_sol
    particleCoords_DM = DMPos
    particleMass_DM  = np.ones(len(np.linalg.norm(DMPos,axis=1))) * 3.07367708626464e-05 * h #1e10 _sol
    particleMass_gas    = gasMass * h #1e0 M_sol
    particleCoords_gas = gasPos
    
    # center coords on subhalo pos
    particleCoords_bar = (particleCoords_bar - subhaloPos_i)*a*h
    particleVels_bar = (particleVels_bar - np.mean(particleVels_bar,axis=0)) * np.sqrt(a)
    particleCoords_DM  = (particleCoords_DM - subhaloPos_i)*a*h
    particleCoords_gas = (particleCoords_gas - subhaloPos_i)*a*h
    
    for i_s in range(len(particleCoords_bar[:,0])):
        if np.linalg.norm(particleCoords_bar[i_s]) == 0:
            particleCoords_bar = np.delete(particleCoords_bar, i_s, axis=0)
            particleVels_bar = np.delete(particleVels_bar, i_s, axis=0)
            particleMass_bar = np.delete(particleMass_bar, i_s)
            break

    for i_dm in range(len(particleCoords_DM[:,0])):
        if np.linalg.norm(particleCoords_DM[i_dm]) == 0:        
            particleCoords_DM = np.delete(particleCoords_DM, i_dm, axis=0)
            particleMass_DM = np.delete(particleMass_DM, i_dm)
            break
            
    #Calculating within the stellar half-mass radius
    r_bar = np.sqrt(np.sum(particleCoords_bar**2,axis=1))
    indsInBounds = np.where((r_bar>0)&(r_bar<=halfmassrad_i))[0]

    J = np.sum(particleMass_bar[indsInBounds].reshape(-1,1)*np.cross(particleCoords_bar[indsInBounds],particleVels_bar[indsInBounds]),axis=0) #kpc*km/s

    spin_vector = make_unit(J)
    #Rotating system to align angular momentum axis and z-axis of the box before calculations 
    z = [[0, 0, 1], [1, 0, 0]]
    rot, rssd, sens = R.align_vectors(z, [spin_vector, [1,0,0]], return_sensitivity=True, weights=[100, .1])
    
    r_DM = np.sqrt(np.sum(particleCoords_DM**2,axis=1))
    r_gas = np.sqrt(np.sum(particleCoords_gas**2,axis=1))
    
    starIndsInDisk = np.where((r_bar>0)&(r_bar<=10*halfmassrad_i))[0]
    DMIndsInDisk = np.where((r_DM>0)&(r_DM<=10*halfmassrad_i))[0]
    gasIndsInDisk = np.where((r_gas>0)&(r_gas<=10*halfmassrad_i))[0]

    rot_starPos = rot.apply(particleCoords_bar[starIndsInDisk])
    rot_starVel = rot.apply(particleVels_bar[starIndsInDisk])
    rot_DMPos = rot.apply(particleCoords_DM[DMIndsInDisk])
    rot_gasPos = rot.apply(particleCoords_gas[gasIndsInDisk])

    massInDisk = particleMass_bar[starIndsInDisk]
    DMInDisk = particleMass_DM[DMIndsInDisk]
    gasInDisk = particleMass_gas[gasIndsInDisk]
    j_z = np.abs(np.cross(rot_starPos,rot_starVel)[:,2])

    pot_star_nbody = agama.Potential(type='Multipole', particles=(rot_starPos, massInDisk), symmetry='None', 
                                    rmin=0.3,rmax=10*halfmassrad_i,gridSizeR=25,lmax=8)
    pot_DM_nbody = agama.Potential(type='Multipole', particles=(rot_DMPos, DMInDisk), symmetry='None', 
                                    rmin=0.3,rmax=10*halfmassrad_i,gridSizeR=25,lmax=8)
    pot_gas_nbody = agama.Potential(type='Multipole', particles=(rot_gasPos, gasInDisk), symmetry='None', 
                                    rmin=0.3,rmax=10*halfmassrad_i,gridSizeR=25,lmax=8)
    joint_pot = agama.Potential(pot_gas_nbody,pot_DM_nbody,pot_star_nbody)    

    BE = (joint_pot.potential(rot_starPos) + 0.5*np.linalg.norm(rot_starVel,axis=1)**2)

    Rcirc = joint_pot.Rcirc(E = BE)
    Tcirc = joint_pot.Tcirc(BE)
    j_circ =  2*np.pi * (Rcirc*3.086e16) * Rcirc / (Tcirc * 3.08633328e+16)

    mass_tot = []
    for i in (range(len(j_z))):
        if j_z[i]/j_circ[i] > 0.7:
            mass_tot.append(massInDisk[i])
    mass_07_fraction = (np.sum(mass_tot)/np.sum(massInDisk))
    
    if mass_07_fraction > 0.5:
        final_diskyIDs.append(subfindID[0])
    NaN_fraction.append(len(np.where(BE > 0)[0])/len(BE))
    total_masses.append(np.sum(massInDisk))

for subfindID in tqdm(main_subfindIDs[883:1244]):
    GrNr = (il.groupcat.loadSingle(basePath, 99, subhaloID=subfindID)['SubhaloGrNr'])

    snapArr,zArr = snapshot_redshift_corr(basePath)

    haloTree = il.lhalotree.loadTree(basePath,99,subfindID,
                                 fields=['SubhaloGrNr','SnapNum','SubhaloPos','SubhaloNumber','SubhaloHalfmassRadType'],onlyMPB=True)

    snap = 75
    haloInd,mpb_snapArr = haloTree['SubhaloGrNr'], haloTree['SnapNum']
    subfindID = haloTree['SubhaloNumber']
    subhaloPos = haloTree['SubhaloPos']
    halfmassrad = haloTree['SubhaloHalfmassRadType']
    h = 0.678

    GrNr_i       = value(haloInd[snap == mpb_snapArr])
    subfindID_i  = value(subfindID[snap == mpb_snapArr])
    subhaloPos_i = value(subhaloPos[snap == mpb_snapArr])
    a = 1/(1+value(zArr[snap==snapArr]))
    halfmassrad_i = value(halfmassrad[snap == mpb_snapArr])[4]*a

    starPos = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Coordinates')
    starMass = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Masses') 
    starVel = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Velocities')
    starU = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Potential')
    DMPos = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,1,fields='Coordinates')
    gasPos = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,0,fields='Coordinates')
    gasMass = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,0,fields='Masses') 

    particleCoords_bar  = starPos
    particleVels_bar    = starVel #km/s
    particleMass_bar    = starMass * h #1e0 M_sol
    particleCoords_DM = DMPos
    particleMass_DM  = np.ones(len(np.linalg.norm(DMPos,axis=1))) * 3.07367708626464e-05 * h #1e10 _sol
    particleMass_gas    = gasMass * h #1e0 M_sol
    particleCoords_gas = gasPos
    
    # center coords on subhalo pos
    particleCoords_bar = (particleCoords_bar - subhaloPos_i)*a*h
    particleVels_bar = (particleVels_bar - np.mean(particleVels_bar,axis=0)) * np.sqrt(a)
    particleCoords_DM  = (particleCoords_DM - subhaloPos_i)*a*h
    particleCoords_gas = (particleCoords_gas - subhaloPos_i)*a*h
    
    for i_s in range(len(particleCoords_bar[:,0])):
        if np.linalg.norm(particleCoords_bar[i_s]) == 0:
            particleCoords_bar = np.delete(particleCoords_bar, i_s, axis=0)
            particleVels_bar = np.delete(particleVels_bar, i_s, axis=0)
            particleMass_bar = np.delete(particleMass_bar, i_s)
            break

    for i_dm in range(len(particleCoords_DM[:,0])):
        if np.linalg.norm(particleCoords_DM[i_dm]) == 0:        
            particleCoords_DM = np.delete(particleCoords_DM, i_dm, axis=0)
            particleMass_DM = np.delete(particleMass_DM, i_dm)
            break
            
    #Calculating within the stellar half-mass radius
    r_bar = np.sqrt(np.sum(particleCoords_bar**2,axis=1))
    indsInBounds = np.where((r_bar>0)&(r_bar<=halfmassrad_i))[0]

    J = np.sum(particleMass_bar[indsInBounds].reshape(-1,1)*np.cross(particleCoords_bar[indsInBounds],particleVels_bar[indsInBounds]),axis=0) #kpc*km/s

    spin_vector = make_unit(J)
    #Rotating system to align angular momentum axis and z-axis of the box before calculations 
    z = [[0, 0, 1], [1, 0, 0]]
    rot, rssd, sens = R.align_vectors(z, [spin_vector, [1,0,0]], return_sensitivity=True, weights=[100, .1])
    
    r_DM = np.sqrt(np.sum(particleCoords_DM**2,axis=1))
    r_gas = np.sqrt(np.sum(particleCoords_gas**2,axis=1))
    
    starIndsInDisk = np.where((r_bar>0)&(r_bar<=10*halfmassrad_i))[0]
    DMIndsInDisk = np.where((r_DM>0)&(r_DM<=10*halfmassrad_i))[0]
    gasIndsInDisk = np.where((r_gas>0)&(r_gas<=10*halfmassrad_i))[0]

    rot_starPos = rot.apply(particleCoords_bar[starIndsInDisk])
    rot_starVel = rot.apply(particleVels_bar[starIndsInDisk])
    rot_DMPos = rot.apply(particleCoords_DM[DMIndsInDisk])
    rot_gasPos = rot.apply(particleCoords_gas[gasIndsInDisk])

    massInDisk = particleMass_bar[starIndsInDisk]
    DMInDisk = particleMass_DM[DMIndsInDisk]
    gasInDisk = particleMass_gas[gasIndsInDisk]
    j_z = np.abs(np.cross(rot_starPos,rot_starVel)[:,2])

    pot_star_nbody = agama.Potential(type='Multipole', particles=(rot_starPos, massInDisk), symmetry='None', 
                                    rmin=0.3,rmax=10*halfmassrad_i,gridSizeR=25,lmax=8)
    pot_DM_nbody = agama.Potential(type='Multipole', particles=(rot_DMPos, DMInDisk), symmetry='None', 
                                    rmin=0.3,rmax=10*halfmassrad_i,gridSizeR=25,lmax=8)
    pot_gas_nbody = agama.Potential(type='Multipole', particles=(rot_gasPos, gasInDisk), symmetry='None', 
                                    rmin=0.3,rmax=10*halfmassrad_i,gridSizeR=25,lmax=8)
    joint_pot = agama.Potential(pot_gas_nbody,pot_DM_nbody,pot_star_nbody)    

    BE = (joint_pot.potential(rot_starPos) + 0.5*np.linalg.norm(rot_starVel,axis=1)**2)

    Rcirc = joint_pot.Rcirc(E = BE)
    Tcirc = joint_pot.Tcirc(BE)
    j_circ =  2*np.pi * (Rcirc*3.086e16) * Rcirc / (Tcirc * 3.08633328e+16)

    mass_tot = []
    for i in (range(len(j_z))):
        if j_z[i]/j_circ[i] > 0.7:
            mass_tot.append(massInDisk[i])
    mass_07_fraction = (np.sum(mass_tot)/np.sum(massInDisk))
    
    if mass_07_fraction > 0.5:
        final_diskyIDs.append(subfindID[0])
    NaN_fraction.append(len(np.where(BE > 0)[0])/len(BE))
    total_masses.append(np.sum(massInDisk))

for subfindID in tqdm(main_subfindIDs[1245:1641]):
    GrNr = (il.groupcat.loadSingle(basePath, 99, subhaloID=subfindID)['SubhaloGrNr'])

    snapArr,zArr = snapshot_redshift_corr(basePath)

    haloTree = il.lhalotree.loadTree(basePath,99,subfindID,
                                 fields=['SubhaloGrNr','SnapNum','SubhaloPos','SubhaloNumber','SubhaloHalfmassRadType'],onlyMPB=True)

    snap = 75
    haloInd,mpb_snapArr = haloTree['SubhaloGrNr'], haloTree['SnapNum']
    subfindID = haloTree['SubhaloNumber']
    subhaloPos = haloTree['SubhaloPos']
    halfmassrad = haloTree['SubhaloHalfmassRadType']
    h = 0.678

    GrNr_i       = value(haloInd[snap == mpb_snapArr])
    subfindID_i  = value(subfindID[snap == mpb_snapArr])
    subhaloPos_i = value(subhaloPos[snap == mpb_snapArr])
    a = 1/(1+value(zArr[snap==snapArr]))
    halfmassrad_i = value(halfmassrad[snap == mpb_snapArr])[4]*a

    starPos = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Coordinates')
    starMass = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Masses') 
    starVel = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Velocities')
    starU = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Potential')
    DMPos = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,1,fields='Coordinates')
    gasPos = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,0,fields='Coordinates')
    gasMass = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,0,fields='Masses') 

    particleCoords_bar  = starPos
    particleVels_bar    = starVel #km/s
    particleMass_bar    = starMass * h #1e0 M_sol
    particleCoords_DM = DMPos
    particleMass_DM  = np.ones(len(np.linalg.norm(DMPos,axis=1))) * 3.07367708626464e-05 * h #1e10 _sol
    particleMass_gas    = gasMass * h #1e0 M_sol
    particleCoords_gas = gasPos
    
    # center coords on subhalo pos
    particleCoords_bar = (particleCoords_bar - subhaloPos_i)*a*h
    particleVels_bar = (particleVels_bar - np.mean(particleVels_bar,axis=0)) * np.sqrt(a)
    particleCoords_DM  = (particleCoords_DM - subhaloPos_i)*a*h
    particleCoords_gas = (particleCoords_gas - subhaloPos_i)*a*h
    
    for i_s in range(len(particleCoords_bar[:,0])):
        if np.linalg.norm(particleCoords_bar[i_s]) == 0:
            particleCoords_bar = np.delete(particleCoords_bar, i_s, axis=0)
            particleVels_bar = np.delete(particleVels_bar, i_s, axis=0)
            particleMass_bar = np.delete(particleMass_bar, i_s)
            break

    for i_dm in range(len(particleCoords_DM[:,0])):
        if np.linalg.norm(particleCoords_DM[i_dm]) == 0:        
            particleCoords_DM = np.delete(particleCoords_DM, i_dm, axis=0)
            particleMass_DM = np.delete(particleMass_DM, i_dm)
            break
            
    #Calculating within the stellar half-mass radius
    r_bar = np.sqrt(np.sum(particleCoords_bar**2,axis=1))
    indsInBounds = np.where((r_bar>0)&(r_bar<=halfmassrad_i))[0]

    J = np.sum(particleMass_bar[indsInBounds].reshape(-1,1)*np.cross(particleCoords_bar[indsInBounds],particleVels_bar[indsInBounds]),axis=0) #kpc*km/s

    spin_vector = make_unit(J)
    #Rotating system to align angular momentum axis and z-axis of the box before calculations 
    z = [[0, 0, 1], [1, 0, 0]]
    rot, rssd, sens = R.align_vectors(z, [spin_vector, [1,0,0]], return_sensitivity=True, weights=[100, .1])
    
    r_DM = np.sqrt(np.sum(particleCoords_DM**2,axis=1))
    r_gas = np.sqrt(np.sum(particleCoords_gas**2,axis=1))
    
    starIndsInDisk = np.where((r_bar>0)&(r_bar<=10*halfmassrad_i))[0]
    DMIndsInDisk = np.where((r_DM>0)&(r_DM<=10*halfmassrad_i))[0]
    gasIndsInDisk = np.where((r_gas>0)&(r_gas<=10*halfmassrad_i))[0]

    rot_starPos = rot.apply(particleCoords_bar[starIndsInDisk])
    rot_starVel = rot.apply(particleVels_bar[starIndsInDisk])
    rot_DMPos = rot.apply(particleCoords_DM[DMIndsInDisk])
    rot_gasPos = rot.apply(particleCoords_gas[gasIndsInDisk])

    massInDisk = particleMass_bar[starIndsInDisk]
    DMInDisk = particleMass_DM[DMIndsInDisk]
    gasInDisk = particleMass_gas[gasIndsInDisk]
    j_z = np.abs(np.cross(rot_starPos,rot_starVel)[:,2])

    pot_star_nbody = agama.Potential(type='Multipole', particles=(rot_starPos, massInDisk), symmetry='None', 
                                    rmin=0.3,rmax=10*halfmassrad_i,gridSizeR=25,lmax=8)
    pot_DM_nbody = agama.Potential(type='Multipole', particles=(rot_DMPos, DMInDisk), symmetry='None', 
                                    rmin=0.3,rmax=10*halfmassrad_i,gridSizeR=25,lmax=8)
    pot_gas_nbody = agama.Potential(type='Multipole', particles=(rot_gasPos, gasInDisk), symmetry='None', 
                                    rmin=0.3,rmax=10*halfmassrad_i,gridSizeR=25,lmax=8)
    joint_pot = agama.Potential(pot_gas_nbody,pot_DM_nbody,pot_star_nbody)    

    BE = (joint_pot.potential(rot_starPos) + 0.5*np.linalg.norm(rot_starVel,axis=1)**2)

    Rcirc = joint_pot.Rcirc(E = BE)
    Tcirc = joint_pot.Tcirc(BE)
    j_circ =  2*np.pi * (Rcirc*3.086e16) * Rcirc / (Tcirc * 3.08633328e+16)

    mass_tot = []
    for i in (range(len(j_z))):
        if j_z[i]/j_circ[i] > 0.7:
            mass_tot.append(massInDisk[i])
    mass_07_fraction = (np.sum(mass_tot)/np.sum(massInDisk))
    
    if mass_07_fraction > 0.5:
        final_diskyIDs.append(subfindID[0])
    NaN_fraction.append(len(np.where(BE > 0)[0])/len(BE))
    total_masses.append(np.sum(massInDisk))

for subfindID in tqdm(main_subfindIDs[1642:1656]):
    GrNr = (il.groupcat.loadSingle(basePath, 99, subhaloID=subfindID)['SubhaloGrNr'])

    snapArr,zArr = snapshot_redshift_corr(basePath)

    haloTree = il.lhalotree.loadTree(basePath,99,subfindID,
                                 fields=['SubhaloGrNr','SnapNum','SubhaloPos','SubhaloNumber','SubhaloHalfmassRadType'],onlyMPB=True)

    snap = 75
    haloInd,mpb_snapArr = haloTree['SubhaloGrNr'], haloTree['SnapNum']
    subfindID = haloTree['SubhaloNumber']
    subhaloPos = haloTree['SubhaloPos']
    halfmassrad = haloTree['SubhaloHalfmassRadType']
    h = 0.678

    GrNr_i       = value(haloInd[snap == mpb_snapArr])
    subfindID_i  = value(subfindID[snap == mpb_snapArr])
    subhaloPos_i = value(subhaloPos[snap == mpb_snapArr])
    a = 1/(1+value(zArr[snap==snapArr]))
    halfmassrad_i = value(halfmassrad[snap == mpb_snapArr])[4]*a

    starPos = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Coordinates')
    starMass = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Masses') 
    starVel = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Velocities')
    starU = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,4,fields='Potential')
    DMPos = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,1,fields='Coordinates')
    gasPos = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,0,fields='Coordinates')
    gasMass = il.snapshot.loadSubhalo(basePath,snap,subfindID_i,0,fields='Masses') 

    particleCoords_bar  = starPos
    particleVels_bar    = starVel #km/s
    particleMass_bar    = starMass * h #1e0 M_sol
    particleCoords_DM = DMPos
    particleMass_DM  = np.ones(len(np.linalg.norm(DMPos,axis=1))) * 3.07367708626464e-05 * h #1e10 _sol
    particleMass_gas    = gasMass * h #1e0 M_sol
    particleCoords_gas = gasPos
    
    # center coords on subhalo pos
    particleCoords_bar = (particleCoords_bar - subhaloPos_i)*a*h
    particleVels_bar = (particleVels_bar - np.mean(particleVels_bar,axis=0)) * np.sqrt(a)
    particleCoords_DM  = (particleCoords_DM - subhaloPos_i)*a*h
    particleCoords_gas = (particleCoords_gas - subhaloPos_i)*a*h
    
    for i_s in range(len(particleCoords_bar[:,0])):
        if np.linalg.norm(particleCoords_bar[i_s]) == 0:
            particleCoords_bar = np.delete(particleCoords_bar, i_s, axis=0)
            particleVels_bar = np.delete(particleVels_bar, i_s, axis=0)
            particleMass_bar = np.delete(particleMass_bar, i_s)
            break

    for i_dm in range(len(particleCoords_DM[:,0])):
        if np.linalg.norm(particleCoords_DM[i_dm]) == 0:        
            particleCoords_DM = np.delete(particleCoords_DM, i_dm, axis=0)
            particleMass_DM = np.delete(particleMass_DM, i_dm)
            break
            
    #Calculating within the stellar half-mass radius
    r_bar = np.sqrt(np.sum(particleCoords_bar**2,axis=1))
    indsInBounds = np.where((r_bar>0)&(r_bar<=halfmassrad_i))[0]

    J = np.sum(particleMass_bar[indsInBounds].reshape(-1,1)*np.cross(particleCoords_bar[indsInBounds],particleVels_bar[indsInBounds]),axis=0) #kpc*km/s

    spin_vector = make_unit(J)
    #Rotating system to align angular momentum axis and z-axis of the box before calculations 
    z = [[0, 0, 1], [1, 0, 0]]
    rot, rssd, sens = R.align_vectors(z, [spin_vector, [1,0,0]], return_sensitivity=True, weights=[100, .1])
    
    r_DM = np.sqrt(np.sum(particleCoords_DM**2,axis=1))
    r_gas = np.sqrt(np.sum(particleCoords_gas**2,axis=1))
    
    starIndsInDisk = np.where((r_bar>0)&(r_bar<=10*halfmassrad_i))[0]
    DMIndsInDisk = np.where((r_DM>0)&(r_DM<=10*halfmassrad_i))[0]
    gasIndsInDisk = np.where((r_gas>0)&(r_gas<=10*halfmassrad_i))[0]

    rot_starPos = rot.apply(particleCoords_bar[starIndsInDisk])
    rot_starVel = rot.apply(particleVels_bar[starIndsInDisk])
    rot_DMPos = rot.apply(particleCoords_DM[DMIndsInDisk])
    rot_gasPos = rot.apply(particleCoords_gas[gasIndsInDisk])

    massInDisk = particleMass_bar[starIndsInDisk]
    DMInDisk = particleMass_DM[DMIndsInDisk]
    gasInDisk = particleMass_gas[gasIndsInDisk]
    j_z = np.abs(np.cross(rot_starPos,rot_starVel)[:,2])

    pot_star_nbody = agama.Potential(type='Multipole', particles=(rot_starPos, massInDisk), symmetry='None', 
                                    rmin=0.3,rmax=10*halfmassrad_i,gridSizeR=25,lmax=8)
    pot_DM_nbody = agama.Potential(type='Multipole', particles=(rot_DMPos, DMInDisk), symmetry='None', 
                                    rmin=0.3,rmax=10*halfmassrad_i,gridSizeR=25,lmax=8)
    pot_gas_nbody = agama.Potential(type='Multipole', particles=(rot_gasPos, gasInDisk), symmetry='None', 
                                    rmin=0.3,rmax=10*halfmassrad_i,gridSizeR=25,lmax=8)
    joint_pot = agama.Potential(pot_gas_nbody,pot_DM_nbody,pot_star_nbody)    

    BE = (joint_pot.potential(rot_starPos) + 0.5*np.linalg.norm(rot_starVel,axis=1)**2)

    Rcirc = joint_pot.Rcirc(E = BE)
    Tcirc = joint_pot.Tcirc(BE)
    j_circ =  2*np.pi * (Rcirc*3.086e16) * Rcirc / (Tcirc * 3.08633328e+16)

    mass_tot = []
    for i in (range(len(j_z))):
        if j_z[i]/j_circ[i] > 0.7:
            mass_tot.append(massInDisk[i])
    mass_07_fraction = (np.sum(mass_tot)/np.sum(massInDisk))
    
    if mass_07_fraction > 0.5:
        final_diskyIDs.append(subfindID[0])
    NaN_fraction.append(len(np.where(BE > 0)[0])/len(BE))
    total_masses.append(np.sum(massInDisk))