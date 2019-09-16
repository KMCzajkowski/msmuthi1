import matplotlib

#matplotlib.use('agg')  # I use agg mode because on some computers it doesn't work without this
import numpy as np
from joblib import Parallel, delayed
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.scattered_field
import smuthi.graphical_output
import smuthi.coordinates as coord
from scipy import io as sio
from scipy import interpolate as sinterp
import numpy.matlib as mlb
#import convfun
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline


# possible future imports
# import matplotlib.pyplot as plt
# import os
# import smuthi.field_expansion as fldexp
# import numpy.matlib as mlb

def set_drude(wl, ep, g, e0):
    """
    Creates a vector y with Drude permittivity
    Input:
    wl - wavelength vector in nm
    ep - resonance energy in eV
    g - gamma in eV
    e0 - eps_inf

    """

    w = 1239.84 / (wl * 10 ** 9)
    y = e0 - ep ** 2 / (w ** 2 + 1j * w * g)
    # re=real(y)
    # ie=imag(y)
    # ndata=1/sqrt(2)*sqrt(re+sqrt(re**2+ie**2))
    # kdata=1/sqrt(2)*sqrt(-re+sqrt(re**2+ie**2))
    return y


def LoadNkData(wl, fname, delim=';', units=1000, omit=1):
    """ Loads nk data from file

    :param wl:
    :param fname:
    :param delim:
    :param units:
    :param omit:
    :return:
    """
    rawd = np.loadtxt(fname, delimiter=delim, skiprows=omit)
    wl0 = rawd[:, 0] * units
    nd = rawd[:, 1]
    kd = rawd[:, 2]
    ni = sinterp.CubicSpline(wl0, nd)
    ki = sinterp.CubicSpline(wl0, kd)
    return ni(wl) + 1j * ki(wl)


def LoadPosFile(fname, cc, r0, z0):
    """ Load position file and rescales it

    :param fname: filename
    :param cc: center-to-center distance
    :param r0: target particle radius
    :param z0:
    :return: position matrix
    """

    posdata = sio.loadmat(fname)
    X = posdata['X'] * cc / posdata['CC'] * r0 / 50
    Y = posdata['Y'] * cc / posdata['CC'] * r0 / 50
    Z = posdata['Y'] * 0 + z0

    X = X[0]
    Y = Y[0]
    Z = Z[0]

    pos = np.concatenate((X, Y, Z))
    pos = pos.reshape((3, 1751))  # beware this evil line!!!
    pos = np.transpose(pos)
    return pos


def SingleParticle(ptype='sph', ref_ind=4, rad=80, height=75, l_max=3, nrank=4):
    if ptype == 'sph':
        positions = [0, 0, -rad]
        particle = smuthi.particles.Sphere(position=positions, refractive_index=ref_ind,
                                           radius=rad, l_max=l_max)
    else:
        positions = [0, 0, -height / 2]
        particle = smuthi.particles.FiniteCylinder(position=positions, refractive_index=ref_ind, cylinder_radius=rad,
                                                   cylinder_height=height, l_max=l_max, m_max=l_max)
        particle.t_matrix_method = {'use discrete sources': True, 'nint': 200, 'nrank': nrank}

    particle_list = [particle]
    return particle_list


def PrepareIdenticalParticles(shape='sph', positions=np.array([0, 0, 100]), ref_ind=1.52, radius=100, height=75,
                              l_max=3, nrank=4):
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


def simsmuthi_norun(wl, parri, par_list, subri=1, topri=1, resolution=3):
    """
    Default smuthi simulation for particles on a substrate
    Specular plane wave incidence with user-specified wavelength.
    Important note: all particles must be made of the same material!
    :param wl: wavelength
    :param parri: refractive index data of the particle
    :param subri: substrate refractive index (default: 1)
    :param topri: refractive index in which the particle is embedded (default: 1)
    :return: tuple with extinction and scattering cross-sections
    """
    for jj in range(len(par_list)):
        par_list[jj].refractive_index = parri

    coord.set_default_k_parallel(wl,
                                 neff_resolution=5e-3,
                                 neff_max=subri + 1)
    two_layers = smuthi.layers.LayerSystem(thicknesses=[0, 0],
                                           refractive_indices=[topri, subri])
    plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=wl,
                                                polar_angle=0,  # from top
                                                azimuthal_angle=0,
                                                polarization=0)  # 0=TE 1=TM
    if len(par_list) < 1:
        simulation = smuthi.simulation.Simulation(layer_system=two_layers,
                                                  particle_list=par_list,
                                                  initial_field=plane_wave,
                                                  solver_type='gmres')
    else:
        simulation = smuthi.simulation.Simulation(layer_system=two_layers,
                                                  particle_list=par_list,
                                                  initial_field=plane_wave,
                                                  solver_type='gmres',
                                                  store_coupling_matrix=False,
                                                  coupling_matrix_lookup_resolution=resolution)
    #simulation.run()
    return simulation

def simsmuthi(wl, parri, par_list, subri=1, topri=1, neff_res=5e-3):
    """
    Default smuthi simulation for particles on a substrate
    Specular plane wave incidence with user-specified wavelength.
    Important note: all particles must be made of the same material!
    :param wl: wavelength
    :param parri: refractive index data of the particle
    :param subri: substrate refractive index (default: 1)
    :param topri: refractive index in which the particle is embedded (default: 1)
    :return: tuple with extinction and scattering cross-sections
    """
    for jj in range(len(par_list)):
        par_list[jj].refractive_index = parri

    coord.set_default_k_parallel(wl,
                                 neff_resolution=neff_res,
                                 neff_max=subri + 1, neff_imag=1e-2)
    two_layers = smuthi.layers.LayerSystem(thicknesses=[0, 0],
                                           refractive_indices=[topri, subri])
    plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=wl,
                                                polar_angle=0,  # from top
                                                azimuthal_angle=0,
                                                polarization=0)  # 0=TE 1=TM
    if len(par_list) < 3:
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
    return simulation


def GetCrossections(simulation):
    angles = np.linspace(0, np.pi, 181)
    # evaluate differential scattering cross section
    dscs = smuthi.scattered_field.scattering_cross_section(initial_field=simulation.initial_field,
                                                           particle_list=simulation.particle_list,
                                                           layer_system=simulation.layer_system)
    cext0 = smuthi.scattered_field.extinction_cross_section(simulation.initial_field, simulation.particle_list,
                                                            simulation.layer_system)
    scat = np.sum(dscs.integral())
    cext = np.real(cext0['top'] + cext0['bottom'])
    return cext, scat


def GetTmatrix(simulation):
    M = simulation.linear_system.master_matrix.linear_operator.A
    Minv = np.linalg.inv(M)
    initcoeff = simulation.particle_list[0].initial_field.coefficients
    scatcoeff = simulation.particle_list[0].scattered_field.coefficients
    tmat = simulation.particle_list[0].t_matrix
    zer = 0*tmat
    if len(simulation.particle_list)>1:
        tmat = None
        #un.listtoarr([tmat,zer,zer,tmat])
    efftmat = np.dot(Minv, tmat)
    return efftmat, initcoeff, scatcoeff


def GetMultipoleSplitCrossesctions(simulation):
    lams = simulation.initial_field.vacuum_wavelength
    Init = simulation.particle_list[0].initial_field.coefficients
    Scat = simulation.particle_list[0].scattered_field.coefficients
    k0 = 2 * np.pi / lams
    k0 = np.transpose(mlb.repmat(k0, len(simulation.particle_list[0].initial_field.coefficients), 1))
    cext0 = -np.pi / k0 ** 2 * np.real(Scat * np.conj(Init))
    csca0 = 1/k0**2 * np.abs(Scat)**2
    return cext0, csca0


def GetForwardAndBackwardScattering(simulation):
    lams = simulation.initial_field.vacuum_wavelength
    Init = simulation.particle_list[0].initial_field.coefficients
    Scat = simulation.particle_list[0].scattered_field.coefficients
    Nm = len(Scat)
    print(Nm)
    k0 = 2 * np.pi / lams
    # k0 = np.transpose(mlb.repmat(k0, len(simulation.particle_list[0].initial_field.coefficients), 1))
    cb0 = 1/k0**2 * np.abs(Scat[0]+Scat[15])**2
    cf0 = 1/k0**2 * np.abs(Scat[0]-Scat[15])**2
    return cb0,cf0


def ParallelSim(waves, nkdata, par_list, subri, topri, ii):
    wl = waves[ii]
    parri = nkdata[ii]
    return simsmuthi(wl, parri, par_list, subri, topri)


def ParallelWavelengthSeries(wl, nkdata, par_list, subri=1, topri=1, num_cores=4):
    res = Parallel(n_jobs=num_cores)(
        delayed(ParallelSim)(wl, nkdata, par_list, subri, topri, ii) for ii in range(len(wl)))
    return res

def findMaxima(wlrange,wyn):
    """
    findMaxima uses interpolation + find_peaks function to find the maxima of arbitrary curve
    :param wlrange: x values of the curve
    :param wyn: y values of the curve
    :return: maxima positions
    """
    wynf = CubicSpline(wlrange, wyn)
    wlrange_dense = np.linspace(np.min(wlrange), np.max(wlrange), 1000)
    wyn_interp = wynf(wlrange_dense)
    peaks, _ = find_peaks(wyn_interp)
    return wlrange_dense[peaks]


if __name__ == '__main__':
    fname = 'Schinke.csv'
    rad = 50
    wl = np.linspace(300, 550, 21)  # wavelength [nm] - must be iterable!
    subri = 2
    num_cores = 4

    nkdata = LoadNkData(wl, fname)
    par_list = SingleParticle(rad=rad)
    res = ParallelWavelengthSeries(wl, nkdata, par_list, subri, num_cores=num_cores)

    Cext = []
    for ii in range(len(wl)):
        cross_sections = GetCrossections(res[ii])
        print(cross_sections)
        Cext.append(cross_sections[0])
