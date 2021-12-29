r'''
Author: Pu Zhichen (hoshishin)
Date: 2021-11-03 09:36:53
LastEditTime: 2021-11-05 08:46:38
LastEditors: Pu Zhichen
Description: 
FilePath: \MCfun\mcfun\core.py

 Multi-collinear approach
'''

import math
from mcfun import lib, xcutils
import multiprocessing
from nptyping import NDArray
import numpy
import typing

NXlib = lib.principle_direction.NX
WXlib = lib.principle_direction.WX

eval_xc_one_direction_DICT = {'LDA' : {0 : xcutils.eval_xc_LDA_energy, 
                                       1 : xcutils.eval_xc_LDA_potential, 
                                       2 : None, 
                                       3 : None, 
                                       4 : None},
                              'GGA' : {0 : xcutils.eval_xc_GGA_energy, 
                                       1 : xcutils.eval_xc_GGA_potential,
                                       2 : None, 
                                       3 : None, 
                                       4 : None},
                              'MGGA': {0 : xcutils.eval_xc_MGGA_energy, 
                                       1 : xcutils.eval_xc_MGGA_potential,
                                       2 : None, 
                                       3 : None, 
                                       4 : None}}

Norder_DICT = {'LDA' : 1,
               'GGA' : 4,
               'MGGA': 6}


def eval_mcol(rho: NDArray[(typing.Any, typing.Any), numpy.float], 
              m: NDArray[(typing.Any, typing.Any, typing.Any), numpy.float], 
              xc_type: str, 
              eval_col:  typing.Callable[[], NDArray[numpy.float]], 
              deriv=1, 
              spin_grid=None)  :    # -> NDArray[(typing.Any,typing.Any),numpy.float]
    """
    Evaluates the multi-collinear functional and its derivatives based on
    the corresponding collinear functional 

    Parameters
    ----------
    rho : np.ndarray
        The density and its derivaties on grids. Rho has the shape (*, Ngrids).
        Depending on the type of XC functional, the leading dimension can be 1
        (LDA), 4 (GGA), or 6 (MGGA):
            [0] the density
            [1] grad_x - the gradient of density along x axis
            [2] grad_y - the gradient of density along y axis
            [3] grad_z - the gradient of density along z axis
            [4] lapl - the laplacian of the density
            [5] tau - the kinetic energy density

    m : np.ndarray
        The magnetization spin vector and its derivaties on grids. m has the shape (3, *, Ngrids).
        The value 3 in the leading dimension corresponds to the 3 components of
        spin density. Depending on the type of XC functional, the second
        dimension can be 1 (LDA), 4 (GGA), or 6 (MGGA):
            [0] the spin density
            [1] grad_x - the gradient of spin density along x axis
            [2] grad_y - the gradient of spin density along y axis
            [3] grad_z - the gradient of spin density along z axis
            [4] lapl - the laplacian of the spin density
            [5] tau - the kinetic energy spin density

    xc_type: string
        The type of XC functional: LDA, GGA, or MGGA.

    eval_col: function
        The function to evaluate the collinear functional.
        The parameters of eval_col should be:
            rho: np.ndarray
                The density rho as defined in eval_mcol.
            s : np.ndarray
                The collinear spin density and its derivaties on grids. s has the shape (*, Ngrids).
                Depending on the type of XC functional, the leading dimension can be 1 (LDA), 4 (GGA), or 6 (MGGA):
                    [0] the spin density
                    [1] grad_x - the gradient of spin density along x axis
                    [2] grad_y - the gradient of spin density along y axis
                    [3] grad_z - the gradient of spin density along z axis
                    [4] lapl - the laplacian of the spin density
                    [5] tau - the kinetic energy spin density
            Returns:
                If deriv == 0 : return exc, vxc
                If deriv == 1 : return exc, vxc, kxc
                If deriv == 2 : return exc, vxc, kxc, fxc
                
                It should be noted that the rho (if exists) should be producted and no more rho is needed.
                Detailed output arrays refer to below, the returns of this function
    
    deriv: integer
        The order of the derivative of exchange-correlation functional
            > 0: energy only
            > 1: energy and potential
            > 2: energy and potential and kernel
            
    spin_grid: default to be None
        If spin_grid is a callable function:
            The method of generating spherical average grids. This functional should
            ` NX, WX = spin_grid() `
            > NX: np.ndarray with the shape (SNgrid, 3), with SNgrid the number of grids of spherical average.
                NX is the coordinates on the unit sphere indicating the projecing directions.
            > WX: np.ndarray with the shape (SNgrid) is the weights of each projecing directions.
            
        Else spin_grid must to be a tuple of (NX, WX). Of course, SNgrid can be 1
        
    Returns:
    ----------
        exc, vxc, fxc, kxc

        where
        
        * exc !! It should be noted that the rho (if exists) should be producted and no more rho is needed.

        * vxc for unrestricted case (vrho, vs):
          | vrho[*, Ngrid]
          | vs[3, *, Ngrid]

        * fxc for unrestricted case ():
          | rho_rho[Ngrid]
          | rho_s[Ngrid]
          | s_s[Ngrid]
          | rho_Nrho[3, Ngrid]
          | Nrho_Nrho[6, Ngrid]
          | s_Nrho[3, Ngrid]
          | rho_Ns[3, Ngrid]
          | s_Ns[3, Ngrid]
          | Nrho_Ns[3, 3, Ngrid]
          | Ns_Ns[6, Ngrid]

        * kxc for unrestricted case:
          | v3rho3[:,4]       = (u_u_u, u_u_d, u_d_d, d_d_d)
          | v3rho2sigma[:,9]  = (u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, d_d_ud, d_d_dd)
          | v3rhosigma2[:,12] = (u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd)
          | v3sigma3[:,10]    = (uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, ud_ud_dd, ud_dd_dd, dd_dd_dd)
            
    """
    
    # ~ init the number of the grids
    Ngrid = rho.shape[-1]    
  
    # ~ Raise some crazy input errors.
    if deriv >= 5:
        raise ValueError("Too high order of the derivatives of Exc")
    
    # ~ Get the  projection axis.
    if spin_grid is None:
        # The default grids of spherical average is using 5810 Lebedev grid.
        # We also provide many different grids ranging from Lebedev, Legendre to Fibonacci grids.
        # 5810 is very very enough to calculate nearly all the systems, so we chose it here.
        # If you want to lower the time or do some tests, you can change the provided grids here.
        NSgrid_default = 5810
        NX = NXlib[NSgrid_default]
        WX = WXlib[NSgrid_default]
    elif callable(spin_grid):
        NX, WX = spin_grid()
    elif type(spin_grid) is tuple:
        NX, WX = spin_grid
    else:
        raise ValueError("spin_grid must be None(using default spherical average scheme,\n"
                         + "callable function, or a tuple of (NX, WX).)")
    SNgrid = WX.shape[-1]
        
    # ~ Get the core function of calculating density functionals.
    eval_xc_one_direction = eval_xc_one_direction_DICT[xc_type.upper()][int(deriv)]
    if eval_xc_one_direction is None:
        raise NotImplementedError(f"{deriv}-th order derivatives of {xc_type} functional is not implemented.")
    
    # ~ init the output array.
    Norder = Norder_DICT[xc_type.upper()]
    if deriv >= 0:
        exc = numpy.zeros((Ngrid))
        vrho = None
        vs = None
        kxc = None
    if deriv >= 1:
        if Norder == 1:
            vrho = numpy.zeros((Ngrid))
            vs = numpy.zeros((3, Ngrid))
        else:
            vrho = numpy.zeros((Norder, Ngrid))
            vs = numpy.zeros((3, Norder, Ngrid))
        kxc = None
    if deriv >= 2:
        pass
    
    # ~ init some parameters in parallel.
    # ncpu: the number of CPU.
    # nsbatch the number of projection directions in one CPU.
    # NX_list [(init, end), ...] contains each CPU batch the initial and end indexes of the NX and WX.
    ncpu = multiprocessing.cpu_count()
    nsbatch = math.ceil(SNgrid/ncpu)
    if SNgrid > 1:
        NX_list = [(i, i+nsbatch) for i in range(0, SNgrid-nsbatch, nsbatch)]
        if NX_list[-1][-1] < SNgrid:
            NX_list.append((NX_list[-1][-1], SNgrid))
    elif  SNgrid == 1:
        NX_list = [(0, 1)]
    else:
        raise ValueError("It should be noted that at least 1 Direction (collinear calculation)\n" +
                         " should be used!")
    pool = multiprocessing.Pool()
    para_results = []
    # ~ parallel run spherical average
    for para in NX_list:
        para_results.append(pool.apply_async(eval_xc_one_direction,
                                             (rho, m, NX, WX, para, eval_col)))
    # ~ finisht the parallel part.
    pool.close()
    pool.join()
    # ~ get the final result
    for result_unpack in para_results:
        result = result_unpack.get()
        if deriv >= 0:
            exc += result[0]
        if deriv >= 1:
            vrho += result[1][0]
            vs += result[1][1]
        if deriv >= 2:
            kxc += result[2]
            
    return exc, (vrho, vs), kxc

    
