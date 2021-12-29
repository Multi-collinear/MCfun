#!/usr/bin/env python
r'''
Author: Pu Zhichen
Date: 2021-11-04 10:56:34
LastEditTime: 2021-11-05 08:46:22
LastEditors: Pu Zhichen
Description: 
FilePath: \MCfun\mcfun\xcutils.py

 Multi-collinear approach
'''

import multiprocessing
from nptyping import NDArray
import numpy
import typing

def wrap_2_eval_col(eval_col_ori: typing.Callable[[], NDArray[numpy.float]], 
                    **kwargs: dict):
    """
    Wrap the original function of evaluating collinear functioanls into a new function contains only
    two parameters, rho and m.

    Parameters
    ----------
    eval_col_ori : The function to evaluate the collinear functional.
        The function may contain some parameters, we don't care in this wrap function
        
    Returns:
    ----------
    eval_col: a callable function contains only the rho and m as basic variables.
            
    """
    def eval_col(rho: NDArray[(typing.Any, typing.Any), numpy.float], 
                 s: NDArray[(typing.Any, typing.Any), numpy.float]):
        """Evaluating the functional derivatives and contains only the rho and m as basic variables.

        Args:
            rho (NDArray[): [description]
            s (NDArray[): [description]

        Returns:
            [type]: [description]
        """
        return eval_col_ori(rho = rho, s = s, **kwargs)
    
    return eval_col


def eval_xc_LDA_energy(rho, m, NX, WX, para, eval_col):
    """Evaluating the energy of LDA functional in a batch of spherical projection directions.

    Parameters
    ----------
    rho: np.ndarray
        The density (Ngrid).
        
    m: np.ndarray
        The magnetization spin vector (3,  Ngrid)

    Returns:
    ----------
    exc_cpu: np.ndarray
        The exchange-corelation vector with the shape (Ngrid)
        
    """
    init, finish = para
    
    exc_cpu = 0.0
    
    for idrct in range(init,finish):
        s = (m[0]*NX[idrct,0]
               + m[1]*NX[idrct,1]
               + m[2]*NX[idrct,2])
        exct, vxct = eval_col(rho, s)
        exc = get_LDA_energy(exct, vxct, s)
        
        exc_cpu+= exc*WX[idrct]
    
    return exc_cpu, None
    

def eval_xc_LDA_potential(rho, m, NX, WX, para, eval_col):
    """Evaluating the energy and potential of LDA functional in a batch of spherical projection directions.

    Parameters
    ----------
    rho: np.ndarray
        The density (Ngrid).
        
    m: np.ndarray
        The magnetization spin vector (3, Ngrid)

    Returns:
    ----------
    exc_cpu: np.ndarray
        The exchange-corelation vector with the shape (Ngrid)
        
    (vrho_cpu, vs_cpu): tuple
        vrho np.ndarray with the shape (Ngrid)
            The rho dependent exchange-corelation potential of a single direction of the multi-collinear approach.
        vs np.ndarray with the shape (3, Ngrid), 
        where the leading dimension is the directions of sigma_x, sigma_y, sigma_z
            The spin dependent exchange-corelation potential of a single direction of the multi-collinear approach.
        
    """
    init, finish = para
    Ngrid = rho.shape[-1]
    
    exc_cpu = 0.0
    vrho_cpu = numpy.zeros((Ngrid))
    vs_cpu = numpy.zeros((3, Ngrid))
    
    for idrct in range(init,finish):
        s = (m[0]*NX[idrct,0]
               + m[1]*NX[idrct,1]
               + m[2]*NX[idrct,2])
        exct, vxct, kxct = eval_col(rho, s)
        exc = get_LDA_energy(exct, vxct, s)
        vrho, vs = get_LDA_potential(vxct, kxct, s)
        
        exc_cpu+= exc*WX[idrct]
        vrho_cpu+= vrho*WX[idrct]
        vs_cpu[0]+= vs*WX[idrct]*NX[idrct,0]
        vs_cpu[1]+= vs*WX[idrct]*NX[idrct,1]
        vs_cpu[2]+= vs*WX[idrct]*NX[idrct,2]
    
    return exc_cpu, (vrho_cpu, vs_cpu)


def get_LDA_energy(exct, vxct, s):
    """Evaluating the energy of LDA functional in a batch of spherical projection directions.

    Parameters
    ----------
    exct: np.ndarray
        The exchange-corelation energy of original LDA function.
        
    vxct: tuple of (vrho, vs), where vrho and vs are np.ndarray with shape (Ngrid)
        The exchange-corelation energy of original LDA function.
        
    s: np.ndarray
        The spin density (Ngrid)

    Returns:
    ----------
    exc = exct + vxct[1]*s : np.ndarray with the shape (Ngrid)
        The exchange-corelation energy of a single direction of the multi-collinear approach.
    """
    return exct + vxct[1]*s


def get_LDA_potential(vxct, kxct, s):
    """Evaluating the potential of LDA functional in a batch of spherical projection directions.

    Parameters
    ----------
    vxct: tuple of (vrho, vs), where vrho and vs are np.ndarray with shape (Ngrid)
        The exchange-corelation energy of original LDA function.
        
    kxct: tuple of (rho_rho, rho_s, s_s, rho_Nrho, Nrho_Nrho, s_Nrho, s_Ns, rho_Ns, Nrho_Ns, Ns_Ns)
        Only the second and third are used in this function
        
    s: np.ndarray
        The spin density (Ngrid)

    Returns:
    ----------
    vrho np.ndarray with the shape (Ngrid)
        The rho dependent exchange-corelation potential of a single direction of the multi-collinear approach.
        
    vs np.ndarray with the shape (Ngrid)
        The spin dependent exchange-corelation potential of a single direction of the multi-collinear approach.
    """
    vrhot, vst = vxct
    rho_st, s_st = kxct[1:3]
    
    vrho = vrhot + s*rho_st
    vs = 2*vst + s*s_st
    
    return vrho, vs
    

def eval_xc_GGA_energy(rho, m, NX, WX, para, eval_col):
    """Evaluating the energy of GGA functional in a batch of spherical projection directions.

    Parameters
    ----------
    rho: np.ndarray
        The density (4,Ngrid).
        
    m: np.ndarray
        The magnetization spin vector (3, 4, Ngrid)

    Returns:
    ----------
    exc_cpu: np.ndarray
        The exchange-corelation vector with the shape (Ngrid)
        
    """
    init, finish = para
    
    exc_cpu = 0.0
    
    for idrct in range(init,finish):
        s = (m[0]*NX[idrct,0]
               + m[1]*NX[idrct,1]
               + m[2]*NX[idrct,2])
        exct, vxct = eval_col(rho, s)
        exc = get_GGA_energy(exct, vxct, s)
        
        exc_cpu+= exc*WX[idrct]
    
    return exc_cpu, None


def eval_xc_GGA_potential(rho, m, NX, WX, para, eval_col):
    """Evaluating the energy of GGA functional in a batch of spherical projection directions.

    Parameters
    ----------
    rho: np.ndarray
        The density (4,Ngrid).
        
    m: np.ndarray
        The magnetization spin vector (3, 4, Ngrid)

    Returns:
    ----------
    exc_cpu: np.ndarray
        The exchange-corelation vector with the shape (Ngrid)
        
    (vrho_cpu, vs_cpu): tuple
        vrho np.ndarray with the shape (4, Ngrid)
            The rho dependent exchange-corelation potential of a single direction of the multi-collinear approach.
        vs np.ndarray with the shape (3, 4, Ngrid), 
        where the leading dimension is the directions of sigma_x, sigma_y, sigma_z
            The spin dependent exchange-corelation potential of a single direction of the multi-collinear approach.    
        
    """
    init, finish = para
    Ngrid = rho.shape[-1]
    
    exc_cpu = 0.0
    vrho_cpu = numpy.zeros((4, Ngrid))
    vs_cpu = numpy.zeros((3, 4, Ngrid))
    
    for idrct in range(init,finish):
        s = (m[0]*NX[idrct,0]
               + m[1]*NX[idrct,1]
               + m[2]*NX[idrct,2])
        exct, vxct, kxct = eval_col(rho, s)
        exc = get_GGA_energy(exct, vxct, s)
        vrho, vs = get_GGA_potential(vxct, kxct, s)
        
        exc_cpu+= exc*WX[idrct]
        vrho_cpu+= vrho*WX[idrct]
        vs_cpu[0]+= vs*WX[idrct]*NX[idrct,0]
        vs_cpu[1]+= vs*WX[idrct]*NX[idrct,1]
        vs_cpu[2]+= vs*WX[idrct]*NX[idrct,2]
    
    return exc_cpu, (vrho_cpu, vs_cpu)


def get_GGA_energy(exct, vxct, s):
    """Evaluating the energy of GGA functional in a batch of spherical projection directions.

    Parameters
    ----------
    exct: np.ndarray
        The exchange-corelation energy of original GGA function.
        
    vxct: tuple of (vrho, vs), where vrho and vs are np.ndarray with shape (4,Ngrid)
        The exchange-corelation energy of original GGA function.
        
    s: np.ndarray
        The spin density (4,Ngrid)

    Returns:
    ----------
    exc = exct + vxct[1]*s : np.ndarray with the shape (Ngrid)
        The exchange-corelation energy of a single direction of the multi-collinear approach.
    """
    vs = vxct[1]
    return exct + s[0]*vs[0] + s[1]*vs[1] + s[2]*vs[2] + s[3]*vs[3]


def get_GGA_potential(vxct, kxct, s):
    """Evaluating the potential of GGA functional in a batch of spherical projection directions.

    Parameters
    ----------
    vxct: tuple of (vrho, vs), where vrho and vs are np.ndarray with shape (4, Ngrid)
        The exchange-corelation energy of original GGA function.
        
    kxct: tuple of (rho_rho, rho_s, s_s, rho_Nrho, Nrho_Nrho, s_Nrho, rho_Ns, s_Ns, Nrho_Ns, Ns_Ns)
        
        
    s: np.ndarray
        The spin density (4, Ngrid)

    Returns:
    ----------
    vrho np.ndarray with the shape (4, Ngrid)
        The rho dependent exchange-corelation potential of a single direction of the multi-collinear approach.
        
    vs np.ndarray with the shape (3, 4, Ngrid), 
    where the leading dimension is the directions of sigma_x, sigma_y, sigma_z
        The spin dependent exchange-corelation potential of a single direction of the multi-collinear approach.
    """
    
    vrhot, vst = vxct
    rho_rho, rho_s, s_s, rho_Nrho, Nrho_Nrho, s_Nrho, rho_Ns, s_Ns, Nrho_Ns, Ns_Ns = kxct
    Ngrid = s.shape[-1]
    
    vrho = numpy.zeros((4, Ngrid))
    vs = numpy.zeros((4, Ngrid))
    
    vrho[0] = vrhot[0] + s[0]*rho_s + s[1]*rho_Ns[0] + s[2]*rho_Ns[1] + s[3]*rho_Ns[2]
    vs[0] = 2*vst[0] + s[0]*s_s + s[1]*s_Ns[0] + s[2]*s_Ns[1] + s[3]*s_Ns[2]
    
    vrho[1] = vrhot[1] + s[0]*s_Nrho[0] + s[1]*Nrho_Ns[0,0] + s[2]*Nrho_Ns[0,1] + s[3]*Nrho_Ns[0,2]
    vrho[2] = vrhot[2] + s[0]*s_Nrho[1] + s[1]*Nrho_Ns[1,0] + s[2]*Nrho_Ns[1,1] + s[3]*Nrho_Ns[1,2]
    vrho[3] = vrhot[3] + s[0]*s_Nrho[2] + s[1]*Nrho_Ns[2,0] + s[2]*Nrho_Ns[2,1] + s[3]*Nrho_Ns[2,2]
    vs[1] = 2*vst[1] + s[0]*s_Ns[0] + s[1]*Ns_Ns[0] + s[2]*Ns_Ns[1] + s[3]*Ns_Ns[2]
    vs[2] = 2*vst[2] + s[0]*s_Ns[1] + s[1]*Ns_Ns[1] + s[2]*Ns_Ns[3] + s[3]*Ns_Ns[4]
    vs[3] = 2*vst[3] + s[0]*s_Ns[2] + s[1]*Ns_Ns[2] + s[2]*Ns_Ns[4] + s[3]*Ns_Ns[5]
    
    return vrho, vs
    
def eval_xc_MGGA_energy(rho, m, NX, WX, para, eval_col):
    """Evaluating the energy of MGGA functional in a batch of spherical projection directions.

    Parameters
    ----------
    rho: np.ndarray
        The density (6, Ngrid).
        
    m: np.ndarray
        The magnetization spin vector (3, 6, Ngrid)

    Returns:
    ----------
    exc_cpu: np.ndarray
        The exchange-corelation vector with the shape (Ngrid)
        
    """
    init, finish = para
    
    exc_cpu = 0.0
    
    for idrct in range(init,finish):
        s = (m[0]*NX[idrct,0]
               + m[1]*NX[idrct,1]
               + m[2]*NX[idrct,2])
        exct, vxct = eval_col(rho, s)
        exc = get_MGGA_energy(exct, vxct, s)
        
        exc_cpu+= exc*WX[idrct]
        
    return exc_cpu, None
    
    
def eval_xc_MGGA_potential(rho, m, NX, WX, para, eval_col):
    """Evaluating the energy and potential of MGGA functional in a batch of spherical projection directions.

    Parameters
    ----------
    rho: np.ndarray
        The density (6, Ngrid).
        
    m: np.ndarray
        The magnetization spin vector (3, 6, Ngrid)

    Returns:
    ----------
    exc_cpu: np.ndarray
        The exchange-corelation vector with the shape (Ngrid)
        
    (vrho_cpu, vs_cpu): tuple
        vrho np.ndarray with the shape (6, Ngrid)
            The rho dependent exchange-corelation potential of a single direction of the multi-collinear approach.
        vs np.ndarray with the shape (3, 6, Ngrid), 
        where the leading dimension is the directions of sigma_x, sigma_y, sigma_z
            The spin dependent exchange-corelation potential of a single direction of the multi-collinear approach.
        
    """
    init, finish = para
    Ngrid = rho.shape[-1]
    
    exc_cpu = 0.0
    vrho_cpu = numpy.zeros((6, Ngrid))
    vs_cpu = numpy.zeros((3, 6, Ngrid))
    
    for idrct in range(init,finish):
        s = (m[0]*NX[idrct,0]
               + m[1]*NX[idrct,1]
               + m[2]*NX[idrct,2])
        exct, vxct, kxct = eval_col(rho, s)
        exc = get_MGGA_energy(exct, vxct, s)
        vrho, vs = get_MGGA_potential(vxct, kxct, s)
        
        exc_cpu+= exc*WX[idrct]
        vrho_cpu+= vrho*WX[idrct]
        vs_cpu[0]+= vs*WX[idrct]*NX[idrct,0]
        vs_cpu[1]+= vs*WX[idrct]*NX[idrct,1]
        vs_cpu[2]+= vs*WX[idrct]*NX[idrct,2]
        
    return exc_cpu, (vrho_cpu, vs_cpu)    


def get_MGGA_energy(exct, vxct, s):
    """Evaluating the energy of MGGA functional in a batch of spherical projection directions.

    Parameters
    ----------
    exct: np.ndarray
        The exchange-corelation energy of original MGGA function.
        
    vxct: tuple of (vrho, vs), where vrho and vs are np.ndarray with shape (6,Ngrid)
        The exchange-corelation energy of original MGGA function.
        
    s: np.ndarray
        The spin density (6,Ngrid)

    Returns:
    ----------
    exc = exct + vxct[1]*s : np.ndarray with the shape (Ngrid)
        The exchange-corelation energy of a single direction of the multi-collinear approach.
    """
    vs = vxct[1]
    return exct + s[0]*vs[0] + s[1]*vs[1] + s[2]*vs[2] + s[3]*vs[3] + s[4]*vs[4] + s[5]*vs[5]


def get_MGGA_potential(vxct, kxct, s):
    """Evaluating the potential of MGGA functional in a batch of spherical projection directions.

    Parameters
    ----------
    vxct: tuple of (vrho, vs), where vrho and vs are np.ndarray with shape (6, Ngrid)
        The exchange-corelation energy of original MGGA function.
        
    kxct: tuple of (rho_rho, rho_s, s_s, rho_Nrho, Nrho_Nrho, s_Nrho, rho_Ns, s_Ns, Nrho_Ns, Ns_Ns, 
        rho_N2rho, rho_N2s, rho_tau, rho_u, s_N2rho, s_N2s, s_tau, s_u,
        Nrho_N2rho, Nrho_N2s, Nrho_tau, Nrho_u, Ns_N2rho, Ns_N2s, Ns_tau, Ns_u,
        N2rho_N2rho, N2rho_N2s, N2rho_tau, N2rho_u, N2s_N2s, N2s_tau, N2s_u,
        tau_tau, tau_u, u_u)
        
        
    s: np.ndarray
        The spin density (6, Ngrid)

    Returns:
    ----------
    vrho np.ndarray with the shape (6, Ngrid)
        The rho dependent exchange-corelation potential of a single direction of the multi-collinear approach.
        
    vs np.ndarray with the shape (3, 6, Ngrid), 
    where the leading dimension is the directions of sigma_x, sigma_y, sigma_z
        The spin dependent exchange-corelation potential of a single direction of the multi-collinear approach.
    """
    vrhot, vst = vxct
    
    rho_rho, rho_s, s_s, rho_Nrho, Nrho_Nrho, s_Nrho, rho_Ns, s_Ns, Nrho_Ns, Ns_Ns, \
        rho_N2rho, rho_N2s, rho_tau, rho_u, s_N2rho, s_N2s, s_tau, s_u,\
        Nrho_N2rho, Nrho_N2s, Nrho_tau, Nrho_u, Ns_N2rho, Ns_N2s, Ns_tau, Ns_u,\
        N2rho_N2rho, N2rho_N2s, N2rho_tau, N2rho_u, N2s_N2s, N2s_tau, N2s_u,\
        tau_tau, tau_u, u_u = kxct
    Ngrid = s.shape[-1]
    
    vrho = numpy.zeros((6, Ngrid))
    vs = numpy.zeros((6, Ngrid))
    
    vrho[0] = vrhot[0] + s[0]*rho_s + s[1]*rho_Ns[0] + s[2]*rho_Ns[1] + s[3]*rho_Ns[2] \
        + s[4]*rho_N2s + s[5]*rho_u
    vs[0] = 2*vst[0] + s[0]*s_s + s[1]*s_Ns[0] + s[2]*s_Ns[1] + s[3]*s_Ns[2] \
        + s[4]*s_N2s + s[5]*s_u
    
    vrho[1] = vrhot[1] + s[0]*s_Nrho[0] + s[1]*Nrho_Ns[0,0] + s[2]*Nrho_Ns[0,1] + s[3]*Nrho_Ns[0,2] \
        + s[4]*Nrho_N2s[0] + s[5]*Nrho_u[0]
    vrho[2] = vrhot[2] + s[0]*s_Nrho[1] + s[1]*Nrho_Ns[1,0] + s[2]*Nrho_Ns[1,1] + s[3]*Nrho_Ns[1,2] \
        + s[4]*Nrho_N2s[1] + s[5]*Nrho_u[1]
    vrho[3] = vrhot[3] + s[0]*s_Nrho[2] + s[1]*Nrho_Ns[2,0] + s[2]*Nrho_Ns[2,1] + s[3]*Nrho_Ns[2,2] \
        + s[4]*Nrho_N2s[2] + s[5]*Nrho_u[2]
    vs[1] = 2*vst[1] + s[0]*s_Ns[0] + s[1]*Ns_Ns[0] + s[2]*Ns_Ns[1] + s[3]*Ns_Ns[2] \
        + s[4]*Ns_N2s[0] + s[5]*Ns_u[0]
    vs[2] = 2*vst[2] + s[0]*s_Ns[1] + s[1]*Ns_Ns[1] + s[2]*Ns_Ns[3] + s[3]*Ns_Ns[4] \
        + s[4]*Ns_N2s[1] + s[5]*Ns_u[1]
    vs[3] = 2*vst[3] + s[0]*s_Ns[2] + s[1]*Ns_Ns[2] + s[2]*Ns_Ns[4] + s[3]*Ns_Ns[5] \
        + s[4]*Ns_N2s[2] + s[5]*Ns_u[2]
        
    vrho[4] = vrhot[4] + s[0]*s_N2rho + s[1]*Ns_N2rho[0] + s[2]*Ns_N2rho[1] + s[3]*Ns_N2rho[2] \
        + s[4]*N2rho_N2s + s[5]*N2rho_u
    vs[4] = 2*vst[4] + s[0]*s_N2s[0] + s[1]*Ns_N2s[0] + s[2]*Ns_N2s[1] + s[3]*Ns_N2s[2] \
        + s[4]*N2s_N2s + s[5]*N2s_u
        
    vrho[5] = vrhot[5] + s[0]*s_tau + s[1]*Ns_tau[0] + s[2]*Ns_tau[1] + s[3]*Ns_tau[2] \
        + s[4]*N2s_tau + s[5]*tau_u
    vs[5] = 2*vst[5] + s[0]*s_u + s[1]*Ns_u[0] + s[2]*Ns_u[1] + s[3]*Ns_u[2] \
        + s[4]*N2s_u + s[5]*u_u
        
    return vrho, vs