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
filename = "./stellar_circs.hdf5" # supplementary stellar circularity information
hf=h5py.File(filename,'r')
circ_subfindID = hf['Snapshot_99']['SubfindID']
circ07 = hf['Snapshot_99']['CircAbove07Frac']
main_subhalos = np.load('/home/tnguser/postprocessing/halocatalogues/' + sim + '_mainSubhalos.npy') #subfindIDs of halos in Neil's catalogue

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

mass_07_fraction = []
for index in tqdm(range(550)):
    subfindID = np.intersect1d(circ_subfindID,main_subhalos)[index]
    GrNr = (il.groupcat.loadSingle(basePath, 99, subhaloID=subfindID)['SubhaloGrNr'])

    snapArr,zArr = snapshot_redshift_corr(basePath)

    haloTree = il.lhalotree.loadTree(basePath,99,subfindID,
                                 fields=['SubhaloGrNr','SnapNum','SubhaloPos','SubhaloNumber','SubhaloHalfmassRadType'],onlyMPB=True)

    snap = 99
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
    particleMass_bar    = starMass / h #1e0 M_sol
    particleCoords_DM = DMPos
    particleMass_DM  = np.ones(len(np.linalg.norm(DMPos,axis=1))) * 3.07367708626464e-05 / h #1e10 _sol
    particleMass_gas    = gasMass / h #1e0 M_sol
    particleCoords_gas = gasPos
    
    # center coords on subhalo pos
    particleCoords_bar = (particleCoords_bar - subhaloPos_i)*a/h
    particleVels_bar = (particleVels_bar - np.mean(particleVels_bar,axis=0)) * np.sqrt(a)
    particleCoords_DM  = (particleCoords_DM - subhaloPos_i)*a/h
    particleCoords_gas = (particleCoords_gas - subhaloPos_i)*a/h

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
    j_circ =  2*np.pi * (Rcirc*3.086e+16) * Rcirc / ((Tcirc * agama.getUnits()['time'])*3.15576e+13)

    mass_tot = []
    for i in (range(len(j_z))):
        if j_z[i]/j_circ[i].value > 0.7:
            mass_tot.append(massInDisk[i])

    mass_07_fraction.append(np.sum(mass_tot)/np.sum(massInDisk))
    
save_mass_07_fractions = save_name+'AV23_G15_shared_mass_fractions.npy'
np.save(save_mass_07_fractions,mass_07_fraction,allow_pickle=True)

indices = []
for i in tqdm(range(550)):
    indices.append(np.where(np.array(hf['Snapshot_99']['SubfindID']) == np.intersect1d(circ_subfindID,main_subhalos)[i])[0])
G15_frac = np.array(hf['Snapshot_99']['CircAbove07Frac'])[indices]

plt.figure(figsize=(4.733,3.55))
plt.scatter(G15_frac, mass_07_fraction,s=1)
plt.xlabel(r'$[F_{*, \mathrm{disky}}]_{\mathrm{G15}}$',fontsize=10)
plt.ylabel(r'$[F_{*, \mathrm{disky}}]_{\mathrm{BFE}}$',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.axhline(0.5,c='r')
plt.axvline(0.5,c='r')
plt.savefig("g15.pdf", format="pdf", bbox_inches="tight")