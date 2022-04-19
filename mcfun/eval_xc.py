#!/usr/bin/env python

'''
Transforms collinear functionals to multi-collinear functionals (the effective
non-collinear functionals)
'''

import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from .LebedevGrid import MakeAngularGrid

def eval_xc_eff(func, rho_tm, deriv=1, spin_samples=770,
                collinear_threshold=None, collinear_samples=200, workers=1):
    '''Multi-collinear effective potential and effective kernel.

    Parameters
    ----------
    func : Function to evaluate collinear functionals.
        The function signature is
            exc, vxc, fxc, ... = func((rho, s), deriv)
        The input rho and s have shape
        * (1, Ngrids) for LDA functionals
        * (4, Ngrids) for GGA functionals where the four variables are
        rho, nabla_x(rho), nabla_y(rho), nabla_z(rho)
        * (5, Ngrids) for meta-GGA functionals where the five variables are
        rho, nabla_x(rho), nabla_y(rho), nabla_z(rho), tau
        The returns of func should have the shape
        * (2, Nvar, Ngrids) for vxc
        * (2, Nvar, 2, Nvar, Ngrids) for fxc
        * (2, Nvar, 2, Nvar, 2, Nvar, Ngrids) for kxc
        The dimension Nvar can be
        * 1 for LDA
        * 4 for GGA: rho, nabla_x(rho), nabla_y(rho), nabla_z(rho)
        * 5 for meta-GGA: rho, nabla_x(rho), nabla_y(rho), nabla_z(rho), tau
        Note: for GGA functionals the required returns have different conventions
        to the returns of libxc or xcfun in functional derivatives. Here func
        needs to return the derivatives to rho, nabla_x(rho), nabla_y(rho),
        nabla_z(rho) while libxc (or xcfun) returns derivatives to rho, sigma_uu,
        sigma_ud, sigma_dd (sigma_ud = nabla(rho_u) dot nabla(rho_d)). Please
        see example 04-xc_wrapper.py for the transformation between the two
        conventions.
    rho_tm : np array with shape (4, Nvar, Ngrids)
        rho_tm[0] is density. rho_tm[1:4] is the magnetization spin vector. Nvar can be
        * 1 for LDA functionals
        * 4 for GGA functionals: rho, nabla_x(rho), nabla_y(rho), nabla_z(rho)
        * 5 for meta-GGA functionals: rho, nabla_x(rho), nabla_y(rho), nabla_z(rho), tau
    deriv : int, optional
        Functional derivatives. The current version (the kernel) supports maximum value 2.
    spin_samples : int, optional
        Number of grid points on spherical surface
    workers : int, optional
        Parallel execution if workers > 1
    collinear_threshold : float, optional
        if specified, filters the points of strongly polarized spins and calls the
        eval_xc_collinear_spin for these points. Recommended value 0.99.
    collinear_samples : int, optional
        Number of samples for eval_xc_collinear_spin.

    Returns
    -------
    XC functional and derivatives for each grid.
    * [exc] if deriv = 0
    * [exc, vxc] if deriv = 1
    * [exc, vxc, fxc] if deriv = 2
    '''
    assert deriv < 3
    if rho_tm.dtype != np.double:
        raise RuntimeError('rho and m must be real')

    if workers == 1:
        return _eval_xc_lebedev(func, rho_tm, deriv, spin_samples,
                                collinear_threshold, collinear_samples)
    else:
        if getattr(func, '__closure__', None):
            warnings.warn(f'Closure {func} cannot be parallelized by multiprocessing module. '
                          'It is recommended to generate the function with functools.partial.')
            executor = ThreadPoolExecutor
        else:
            executor = ProcessPoolExecutor

        ngrids = rho_tm[0].shape[-1]
        with executor(max_workers=workers) as ex:
            futures = []
            for p0, p1 in _prange(0, ngrids, ngrids//workers+1):
                futures.append(ex.submit(_eval_xc_lebedev, func,
                                         rho_tm[...,p0:p1], deriv, spin_samples))
            results = [f.result() for f in futures]
        return [None if x[0] is None else np.concatenate(x, axis=-1) for x in zip(*results)]

def eval_xc_collinear_spin(func, rho_tm, deriv, spin_samples):
    '''Multi-collinear functional derivatives for collinear spins

    Parameters
    ----------
    func : Function to evaluate collinear functionals.
        The function signature is
            exc, vxc, fxc, ... = func((rho, s), deriv)
        The input rho and s have shape
        * (1, Ngrids) for LDA functionals
        * (4, Ngrids) for GGA functionals where the four variables are
        rho, nabla_x(rho), nabla_y(rho), nabla_z(rho)
        * (5, Ngrids) for meta-GGA functionals where the five variables are
        rho, nabla_x(rho), nabla_y(rho), nabla_z(rho), tau
        The returns of func should have the shape
        * (2, Nvar, Ngrids) for vxc
        * (2, Nvar, 2, Nvar, Ngrids) for fxc
        * (2, Nvar, 2, Nvar, 2, Nvar, Ngrids) for kxc
        The dimension Nvar can be
        * 1 for LDA
        * 4 for GGA: rho, nabla_x(rho), nabla_y(rho), nabla_z(rho)
        * 5 for meta-GGA: rho, nabla_x(rho), nabla_y(rho), nabla_z(rho), tau
        Note: for GGA functionals the required returns have different conventions
        to the returns of libxc or xcfun in functional derivatives. Here func
        needs to return the derivatives to rho, nabla_x(rho), nabla_y(rho),
        nabla_z(rho) while libxc (or xcfun) returns derivatives to rho, sigma_uu,
        sigma_ud, sigma_dd (sigma_ud = nabla(rho_u) dot nabla(rho_d)).
    rho_tm : np array with shape (4, Nvar, Ngrids)
        rho_tm[0] is density. rho_tm[1:4] is the magnetization spin vector. Nvar can be
        * 1 for LDA functionals
        * 4 for GGA functionals: rho, nabla_x(rho), nabla_y(rho), nabla_z(rho)
        * 5 for meta-GGA functionals: rho, nabla_x(rho), nabla_y(rho), nabla_z(rho), tau
    deriv : int, optional
        Functional derivatives. The current version (the kernel) supports maximum value 2.
    spin_samples : int, optional
        Number of grid points on principal axis

    Returns
    -------
    XC functional and derivatives for each grid.
    * [exc] if deriv = 0
    * [exc, vxc] if deriv = 1
    * [exc, vxc, fxc] if deriv = 2
    '''
    ngrids = rho_tm[0].shape[-1]
    # samples on z=cos(theta) and their weights between [0, 1]
    sgridz, weights = _make_paxis_samples(spin_samples)
    blksize = int(np.ceil(1e5 / ngrids)) * 8

    if rho_tm.ndim == 2:
        nvar = 1
    else:
        nvar = rho_tm.shape[1]

    rho_ts = _project_spin_paxis(rho_tm)
    m = rho_tm[1:].reshape(3, nvar, ngrids)
    s = rho_ts[1]

    # TODO: filter s, nabla(s) ~= 0
    omega[s==0] = 0
    omega[s!=0] = m[s!=0]/s[s!=0]
    xc_orig = func(rho_ts, deriv)
    exc_eff = xc_orig[0]

    omega = omega.reshape(3, ngrids)
    if deriv > 0:
        vxc = xc_orig[1].reshape(2, nvar, ngrids)
        vxc_eff = np.vstack((vxc[:1], np.einsum('xg,rg->rxg', vxc[1], omega)))

    if deriv > 1:
        # spin-conserve part
        fxc = xc_orig[2].reshape(2, nvar, 2, nvar, ngrids)
        fxc_eff = np.empty((4, nvar, 4, nvar, ngrids))
        fxc_eff[0,:,0] = fxc[0,:,0]
        fz1 = np.einsum('xyg,rg->rxyg', fxc[1,:,0], omega)
        fxc_eff[1:,:,0 ] = fz1
        fxc_eff[0 ,:,1:] = fz1.transpose(1,0,2,3)
        tmp = np.einsum('xyg,rg->rxyg', fxc[1,:,1,:], omega)
        fxc_eff[1:,:,1:] = np.einsum('rxyg,sg->rxsyg', tmp, omega)

        # spin-flip part
        fxc_sf = 0
        for p0, p1 in _prange(0, weights.size, blksize):
            rho = _project_spin_paxis(rho_tm, sgridz[p0:p1])
            fxc = func(rho, deriv)[2]
            fxc = fxc.reshape(2, nvar, 2, nvar, ngrids, p1 - p0)
            # only considers the xx+yy part
            fxc_sf += fxc[1,:,1].dot(weights[p0:p1])

        for i in range(1, 4):
            fxc_eff[i,:,i] += fxc_sf
        tmp = np.einsum('xyg,rg->rxyg', fxc_sf, omega)
        fxc_eff[1:,:,1:] -= np.einsum('rxyg,sg->rxsyg', tmp, omega)

    ret = [exc_eff]
    if deriv > 0:
        ret.append(vxc_eff)
    if deriv > 1:
        ret.append(fxc_eff)
    if deriv > 2:
        raise NotImplementedError
    return ret

def _eval_xc_lebedev(func, rho_tm, deriv, spin_samples,
                     collinear_threshold=None, collinear_samples=200):
    '''Multi-collinear effective potential and effective kernel with projection
    samples on spherical surface (the Lebedev grid samples)
    '''
    ngrids = rho_tm[0].shape[-1]
    sgrids, weights = _make_sph_samples(spin_samples)
    blksize = int(np.ceil(5e4 / ngrids)) * 8

    if rho_tm.ndim == 2:
        nvar = 1
    else:
        nvar = rho_tm.shape[1]

    exc_eff = vxc_eff = fxc_eff = 0
    for p0, p1 in _prange(0, weights.size, blksize):
        nsg = p1 - p0
        p_sgrids = sgrids[p0:p1]
        p_weights = weights[p0:p1]
        rho = _project_spin_sph(rho_tm, p_sgrids)
        xc_orig = func(rho, deriv+1)

        exc = xc_orig[0].reshape(ngrids, nsg)
        vxc = xc_orig[1].reshape(2, nvar, ngrids, nsg)

        s = rho[1].reshape(nvar,ngrids,nsg)
        rho_pure = rho[0].reshape(nvar,ngrids,nsg)[0]
        exc[rho_pure == 0] = 0
        exc[rho_pure != 0] /= rho_pure[rho_pure != 0]
        exc += np.einsum('xgo,xgo->go', vxc[1], s)
        exc_eff += np.einsum('go,o->g', exc, p_weights)

        if deriv > 0:
            fxc = xc_orig[2].reshape(2, nvar, 2, nvar, ngrids, nsg)
            # vs * 2 + s*f_s_st
            vxc[1] *= 2
            vxc += np.einsum('xbygo,xgo->bygo', fxc[1], s)
            c_tm = _ts2tm_transformation(p_sgrids)
            cw_tm = c_tm * p_weights
            vxc_eff += np.einsum('rao,axgo->rxg', cw_tm, vxc)

        if deriv > 1:
            kxc = xc_orig[3].reshape(2, nvar, 2, nvar, 2, nvar, ngrids, nsg)
            fxc[1,:,1] *= 3
            fxc[0,:,1] *= 2
            fxc[1,:,0] *= 2
            fxc += np.einsum('xbyczgo,xgo->byczgo', kxc[1], s)
            fxc = np.einsum('rao,axbygo->rxbygo', c_tm, fxc)
            fxc_eff += np.einsum('sbo,rxbygo->rxsyg', cw_tm, fxc)

    # Strongly spin-polarized points (rho ~= |m|) can be considered as collinear spins
    if collinear_threshold is not None:
        rho, s = _project_spin_paxis(rho_tm.reshape(4, nvar, ngrids)[:,0])
        cs_idx = np.where(s >= rho * collinear_threshold)[0]
        if cs_idx.size > 0:
            xc_cs = eval_xc_collinear_spin(func, rho_tm[...,cs_idx], deriv,
                                           collinear_samples)
            exc_eff[...,cs_idx] = xc_cs[0]
            if deriv > 0:
                vxc_eff[...,cs_idx] = xc_cs[1]
            if deriv > 1:
                fxc_eff[...,cs_idx] = xc_cs[2]

    ret = [exc_eff]
    if deriv > 0:
        ret.append(vxc_eff)
    if deriv > 1:
        ret.append(fxc_eff)
    if deriv > 2:
        raise NotImplementedError
    return ret

def _make_sph_samples(spin_samples):
    '''Integration samples on spherical surface'''
    ang_grids = MakeAngularGrid(spin_samples)
    directions = ang_grids[:,:3].copy(order='F')
    weights = ang_grids[:,3].copy()
    return directions, weights

def _prange(start, end, step):
    '''Partitions range into segments: i0:i1, i1:i2, i2:i3, ...'''
    if start < end:
        for i in range(start, end, step):
            yield i, min(i+step, end)

def _project_spin_sph(rho_tm, sgrids):
    '''Projects spin onto spherical surface'''
    rho = rho_tm[0]
    m = rho_tm[1:]
    nsg = sgrids.shape[0]
    ngrids = rho.shape[-1]
    if rho_tm.ndim == 2:
        rho_ts = np.empty((2, ngrids, nsg))
        rho_ts[0] = rho[:, np.newaxis]
        rho_ts[1] = np.einsum('mg,om->go', m, sgrids)
        rho_ts = rho_ts.reshape(2, ngrids*nsg)
    else:
        nvar = rho_tm.shape[1]
        rho_ts = np.empty((2, nvar, ngrids, nsg))
        rho_ts[0] = rho[:, :, np.newaxis]
        rho_ts[1] = np.einsum('mxg,om->xgo', m, sgrids)
        rho_ts = rho_ts.reshape(2, nvar, ngrids*nsg)
    return rho_ts

def _ts2tm_transformation(sgrids):
    '''
    Transformation that projects v_ts(rho,s) to rho/m representation (rho,mx,my,mz)
    '''
    nsg = sgrids.shape[0]
    c_tm = np.zeros((4, 2, nsg))
    c_tm[0,0] = 1
    c_tm[1:,1] = sgrids.T
    return c_tm

def _make_paxis_samples(spin_samples):
    '''Samples on principal axis between [0, 1]'''
    rt, wt = np.polynomial.legendre.leggauss(spin_samples)
    rt = rt * .5 + .5
    wt *= .5  # normalized to 1
    return rt, wt

def _project_spin_paxis(rho_tm, sgridz=None):
    '''Projects spins onto the principal axis'''
    rho = rho_tm[0]
    m = rho_tm[1:]
    s = np.linalg.norm(m, axis=0)
    if sgridz is None:
        rho_ts = np.stack([rho, s])
    else:
        ngrids = rho.shape[-1]
        nsg = sgridz.shape[0]
        if rho_tm.ndim == 2:
            rho_ts = np.empty((2, ngrids, nsg))
            rho_ts[0] = rho[:,np.newaxis]
            rho_ts[1] = s[:,np.newaxis] * sgridz
            rho_ts = rho_ts.reshape(2, ngrids * nsg)
        else:
            nvar = rho_tm.shape[1]
            rho_ts = np.empty((2, nvar, ngrids, nsg))
            rho_ts[0] = rho[:,:,np.newaxis]
            rho_ts[1] = s[:,:,np.newaxis] * sgridz
            rho_ts = rho_ts.reshape(2, nvar, ngrids * nsg)
    return rho_ts
