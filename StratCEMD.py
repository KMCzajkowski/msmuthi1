import mysmuthi as ms
from numpy import matlib as mlb
from scipy import io as sio
import smuthi.particle_coupling as coup
import numpy as np
from numpy.linalg import inv
import pyCEMD
import smuthi.t_matrix as tmt
import matplotlib.pyplot as plt
from smuthi import layers
import smuthi.coordinates as coord
import smuthi.initial_field as init_field


def layercoupling(wl, par_list, layers, M):
    k0 = 2 * np.pi / wl
    c = -6 * 1j * k0 ** (-3) / 4
    N_particles = len(par_list)
    blocksize = 6
    Wr = np.zeros((N_particles * blocksize, N_particles * blocksize), dtype=complex)
    for ii in range(N_particles):
        for jj in range(N_particles):
            Wr[ii * blocksize:blocksize * (ii + 1),
            jj * blocksize:blocksize * (jj + 1)] = coup.layer_mediated_coupling_block(wl, par_list[ii], par_list[jj],
                                                                                      layers, 'default')
    Gr = change_basis(Wr, M, c)
    return Gr


def particlecoupling(wl, pos):
    k0 = 2 * np.pi / wl
    N_particles = pos.shape[0]
    blocksize = 6
    hbr = int(blocksize / 2)
    A = np.zeros((N_particles * blocksize, N_particles * blocksize), dtype=complex)
    for ii in range(N_particles):
        for jj in range(N_particles):
            if not (ii == jj):
                A0 = np.zeros((blocksize, blocksize), dtype=complex)
                g0 = pyCEMD.gij_matrix(k0, pos[ii, :], pos[jj, :])
                a0 = pyCEMD.Aij_matrix(k0, pos[ii, :], pos[jj, :])
                A0[0:hbr, 0:hbr] = a0
                A0[0:hbr, hbr:blocksize] = g0
                A0[hbr:blocksize, 0:hbr] = g0
                A0[hbr:blocksize, hbr:blocksize] = a0
                A[ii * blocksize:blocksize * (ii + 1),
                jj * blocksize:blocksize * (jj + 1)] = A0
    return A


def inversepolarizabilities(par_list, wl, identical=1, niS=1):
    N_particles = len(par_list)
    k0 = 2 * np.pi / wl
    c = -6 * 1j * k0 ** (-3) / 4
    M = conversionmatrix(N_particles, c)
    t_mat0 = tmt.t_matrix(wl, niS, par_list[0])
    t_mat=blockdiagonal(t_mat0,N_particles)
    alfinv=change_basis(inv(t_mat),M,c)
    #alf = change_basis(t_mat0, M, c)
    #alfinv0 = inv(alf)
    #alfinv = blockdiagonal(alfinv0, N_particles)
    return alfinv


def initial_field(wl, par_list, layer_system):
    plane_wave = init_field.PlaneWave(vacuum_wavelength=wl,
                                      polar_angle=0,
                                      azimuthal_angle=0,
                                      polarization=0)  # 0=TE 1=TM
    N_particles = len(par_list)
    k0 = 2 * np.pi / wl
    c = -6 * 1j * k0 ** (-3) / 4
    init0 = plane_wave.spherical_wave_expansion(par_list[0], layer_system).coefficients
    M = conversionmatrix(1, c)
    E_inc = 1 / c * M @ init0 / np.sqrt(3)
    init = mlb.repmat(E_inc, 1, N_particles)
    return init[0]


def solve(alfinv, A, Gr, E_inc):
    Meq = (alfinv - A - Gr)
    Minv = inv(Meq)
    p = Minv @ E_inc
    return p


def extinction_cross_section(wl, E_inc, p):
    k0 = 2 * np.pi / wl
    c_ext = (4 * np.pi * k0) * np.sum(np.imag(np.dot(np.conj(E_inc), p)))
    return c_ext


def conversionmatrix(N, c):
    Mtst0 = 0.5 * np.array([[1j, 0, 1j], [1, 0, -1], [0, -np.sqrt(2), 0]], dtype=complex)
    zer = np.array(np.zeros((3, 3)))
    Mtst1 = np.hstack((Mtst0, zer))
    Mtst2 = np.hstack((zer, Mtst0))
    Mtst = np.vstack((Mtst1, Mtst2))
    M = c * blockdiagonal(Mtst, N)
    return M


def blockdiagonal(M, N):
    # assuming square M
    Nbl = M.shape[0]
    Mbig = np.zeros((Nbl * N, Nbl * N), dtype=complex)
    M0 = np.zeros(M.shape, dtype=complex)
    for ii in range(N):
        for jj in range(N):
            if ii == jj:
                Mbig[ii * Nbl: (ii + 1) * Nbl, jj * Nbl: (jj + 1) * Nbl] = M
            else:
                Mbig[ii * Nbl:(ii + 1) * Nbl, jj * Nbl:(jj + 1) * Nbl] = M0
    return Mbig


def change_basis(Wr, M, c):
    Gr = 1 / c * M @ Wr @ inv(M)
    return Gr


def findmax(lam, cabs, steps=5, degree=4):
    """ finds maximum!

    :param lam:
    :param cabs:
    :return:
    """
    indmax = np.argmax(cabs)
    p = np.poly1d(np.polyfit(lam[indmax - steps:indmax + steps], cabs[indmax - steps:indmax + steps], degree))
    lamdet = np.linspace(lam[indmax - steps], lam[indmax + steps], 1000)
    plt.plot(lamdet, p(lamdet))
    plt.plot(lam, cabs)
    indmax = np.argmax(p(lamdet))

    return lamdet[indmax]

if __name__=="__main__":
    print('StratCDA by K.M.C.')
    # sample simulation for two spheres with refractive index of 4 at 370 nm wavelength placed on a substrate
    # with refractive index equal to 2
    fname = 'Schinke.csv'
    rad = 45  # sphere radii - assuming identical particles
    wl = np.array([370])
    wl = np.linspace(250, 500, 21) # another example declaration of wavelengths
    subri = 2  # substrate refractive index
    # nkdata = ms.LoadNkData(wl, fname) # use if you want to import Si data from Schinke
    nkdata = 0 * wl + 4  # particle refractive index
    CC = 7000
    pos = np.array([[0, 0, -rad], [CC * rad, CC * rad, -rad]])  # list of particle positions
    # pos = ms.LoadPosFile(posname, CC, rad, -rad) # loads position from posname and scales it to match rad
    N_particles = pos.shape[0]
    cext_tot=[]
    for ii in range(len(wl)):
        # set_default_k_parallel is extremely tricky function from Amos. Use with care.
        coord.set_default_k_parallel(wl[ii],
                                     neff_resolution=5e-3,
                                     neff_max=subri + 1)
        two_layers = layers.LayerSystem(thicknesses=[0, 0],
                                        refractive_indices=[1, subri])
        par_list = ms.PrepareIdenticalParticles(positions=pos, radius=rad, l_max=1, ref_ind=nkdata[ii])
        k0 = 2*np.pi / wl[ii]
        c = -6 * 1j * k0 ** (-3) / 4
        M = conversionmatrix(N_particles, c)  # not too smart!
        Sr=layercoupling(wl[ii], par_list, two_layers, M)
        S=particlecoupling(wl[ii], pos)
        alfinv=inversepolarizabilities(par_list, wl[ii])
        Einc=initial_field(wl[ii], par_list, two_layers)
        p=solve(alfinv, S, Sr, Einc)
        cext=extinction_cross_section(wl[ii], Einc, p)
        cext_tot.append(cext)
        #print(Einc)
        #print(alfinv)
        #print(S)
        #print(Sr)
        #print(M)
    plt.plot(wl,cext_tot)
    plt.show()