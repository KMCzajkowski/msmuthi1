import matplotlib

matplotlib.use('agg')
from joblib import Parallel, delayed
import numpy as np
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.scattered_field
import smuthi.graphical_output
import matplotlib.pyplot as plt
import os
import smuthi.coordinates as coord
import smuthi.field_expansion as fldexp
import numpy.matlib as mlb
from scipy import io as sio
from scipy import interpolate as sinterp


def LoadNkData(wl, fname, delim=';', units=1000, omit=1):
    rawd = np.loadtxt(fname, delimiter=delim, skiprows=omit)
    wl0 = rawd[:, 0] * units
    nd = rawd[:, 1]
    kd = rawd[:, 2]
    ni = sinterp.CubicSpline(wl0, nd)
    ki = sinterp.CubicSpline(wl0, kd)
    return ni(wl) + 1j * ki(wl)


def LoadPosFile(fname, cc, r0, z0):
    posdata = sio.loadmat(fname)
    X = posdata['X'] * cc / posdata['CC'] * r0 / 50
    Y = posdata['Y'] * cc / posdata['CC'] * r0 / 50
    Z = posdata['Y'] * 0 + z0

    X = X[0]
    Y = Y[0]
    Z = Z[0]

    pos = np.concatenate((X, Y, Z))
    pos = pos.reshape((3, 1751))
    pos = np.transpose(pos)
    return pos


def PrepareIdenticalParticles(shape='sph', positions=np.array([0, 0, 100]), ref_ind=1.52, radius=100, height=75,
                              l_max=2, nrank=4):
    if len(positions.shape) > 1:
        N = positions.shape[0]
    else:
        N = 1
        positions = np.array([0, 0, radius])
    print('N=' + str(N))

    particle_list = []
    for ii in range(N):
        if N > 1:
            pos = list(positions[ii])
        else:
            pos = list(positions)
        if shape == 'sph':
            particle = smuthi.particles.Sphere(position=pos, refractive_index=ref_ind, radius=radius, l_max=l_max)
        else:
            particle = smuthi.particles.FiniteCylinder(position=pos, refractive_index=ref_ind, cylinder_radius=radius,
                                                       cylinder_height=height, l_max=l_max, m_max=l_max)
            particle.t_matrix_method = {'use discrete sources': True, 'nint': 200, 'nrank': nrank}
        particle_list.append(particle)
    return particle_list


def simsmuthi(ii, lams, nkdata, fname, rad, subri):
    wl = lams[ii]
    parri = nkdata[ii]
    angles = np.linspace(0, np.pi, 181)
    z0 = -rad
    pos = LoadPosFile(fname, cc, rad, z0)
    par_list = PrepareIdenticalParticles(positions=pos, ref_ind=parri, radius=rad)

    coord.set_default_k_parallel(wl,
                                 neff_resolution=5e-3,
                                 neff_max=subri + 1)
    two_layers = smuthi.layers.LayerSystem(thicknesses=[0, 0],
                                           refractive_indices=[1, subri])
    plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=wl,
                                                polar_angle=0,  # from top
                                                azimuthal_angle=0,
                                                polarization=0)  # 0=TE 1=TM
    if len(par_list) == 1:
        simulation = smuthi.simulation.Simulation(layer_system=two_layers,
                                                  particle_list=par_list,
                                                  initial_field=plane_wave,
                                                  solver_type='gmres')
    else:
        simulation = smuthi.simulation.Simulation(layer_system=two_layers,
                                                  particle_list=par_list,
                                                  initial_field=plane_wave,
                                                  solver_type='KCSolver',
                                                  store_coupling_matrix=False,
                                                  coupling_matrix_lookup_resolution=3)
    simulation.run()
    # evaluate differential scattering cross section
    dscs = smuthi.scattered_field.scattering_cross_section(initial_field=plane_wave,
                                                           particle_list=par_list,
                                                           layer_system=two_layers, polar_angles=angles)
    cext0 = smuthi.scattered_field.extinction_cross_section(plane_wave, par_list, two_layers)
    scat = dscs.integral()
    cext = cext0['top'] + cext0['bottom']
    return cext, scat

num_cores = 1
fname = 'tt_D100_N1751.mat'
ccvec = np.arange(2.5, 10.5, 0.5)
cc = [5]
r0 = 50
subri = 1
#lams = np.linspace(350, 500, 61)
lams = [400]
nkdata = LoadNkData(lams, 'Schinke.csv')


tst = simsmuthi(0,lams,nkdata,fname,r0,subri)
#for cc in ccvec:
#    matfile = 'output' + str(cc) + '.mat'
#    res = Parallel(n_jobs=num_cores)(delayed(simsmuthi)(ii, lams, nkdata, fname, r0, subri) for ii in range(len(lams)))
#    Cext = [item[0] for item in res]
#    Cscat = [item[1] for item in res]
#    a = {'wl': lams, 'cext': Cext, 'cscat': Cscat}
#    sio.savemat(matfile, a)
