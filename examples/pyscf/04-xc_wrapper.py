#!/usr/bin/env python

'''
A wrapper to transform eval_xc function in pyscf using the mcfun eval_xc_eff
'''


import functools
import numpy as np
from pyscf.dft.libxc import eval_xc
from mcfun import eval_xc_eff

def mcfun_eval_xc_wrapper(xc_code, xctype):
    '''Wrapper to generate the eval_xc function required by mcfun'''
    return functools.partial(_mcfun_eval_xc, xc_code, xctype=xctype)

def _mcfun_eval_xc(xc_code, rho, deriv, xctype):
    t, s = rho
    if xctype == 'MGGA':
        # Padding for laplacian
        ngrids = rho.shape[-1]
        rhop = np.empty((2, 6, ngrids))
        rhop[0,:4] = (t[:4] + s[:4]) * .5
        rhop[1,:4] = (t[:4] - s[:4]) * .5
        rhop[:,4] = 0
        rhop[0,5] = (t[4] + s[4]) * .5
        rhop[1,5] = (t[4] - s[4]) * .5
    else:
        rhop = np.empty_like(rho)
        rhop[0] = (t + s) * .5
        rhop[1] = (t - s) * .5

    exc, vxc, fxc, kxc = eval_xc(xc_code, rhop, spin=1, deriv=deriv)

    if deriv > 2:
        kxc = transform_kxc(rhop, fxc, kxc, xctype, spin=1)
        kxc = ud2ts(kxc)
    if deriv > 1:
        fxc = transform_fxc(rhop, vxc, fxc, xctype, spin=1)
        fxc = ud2ts(fxc)
    if deriv > 0:
        vxc = transform_vxc(rhop, vxc, xctype, spin=1)
        vxc = ud2ts(vxc)
    return exc, vxc, fxc, kxc

################## xc_deriv ####################
# Functions to transform pyscf libxc.eval_xc output to standard form.
# Copied from pyscf/dft/xc_deriv.py
def transform_vxc(rho, vxc, xctype, spin=0):
    r'''
    Transform libxc functional derivatives to the derivative tensor of
    parameters in rho:
        [[density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a],
         [density_b, (nabla_x)_b, (nabla_y)_b, (nabla_z)_b, tau_b]].
    The output tensor has the shape:
        * spin polarized
            LDA : [2,1,N]
            GGA : [2,4,N]
            MGGA: [2,6,N]
        * spin unpolarized
            LDA : [1,N]
            GGA : [4,N]
            MGGA: [6,N]
    '''
    rho = np.asarray(rho, order='C')
    if xctype == 'GGA':
        order = 1
        nvar = 4
        fr = vxc[0].T
        fg = vxc[1].T
    elif xctype == 'MGGA':
        order = 2
        nvar = 5
        fr = vxc[0].T
        fg = vxc[1].T
        ft = vxc[3].T
    else:  # LDA
        order = 0
        nvar = 1
        fr = vxc[0].T

    ngrids = rho.shape[-1]
    if spin == 1:
        vp = np.empty((2,nvar, ngrids))
        vp[:,0] = fr
        if order > 0:
            vp[:,1:4] = np.einsum('abg,bxg->axg', _stack_fg(fg), rho[:,1:4])
        if order > 1:
            vp[:,4] = ft
    else:
        vp = np.empty((nvar, ngrids))
        vp[0] = fr
        if order > 0:
            vp[1:4] = 2 * fg * rho[1:4]
        if order > 1:
            vp[4] = ft
    return vp

def transform_fxc(rho, vxc, fxc, xctype, spin=0):
    r'''
    Transform libxc functional derivatives to the derivative tensor of
    parameters in rho:
        [[density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a],
         [density_b, (nabla_x)_b, (nabla_y)_b, (nabla_z)_b, tau_b]].
    The output tensor has the shape:
        * spin polarized
            LDA : [2,1,2,1,N]
            GGA : [2,4,2,4,N]
            MGGA: [2,6,2,6,N]
        * spin unpolarized
            LDA : [1,1,N]
            GGA : [4,4,N]
            MGGA: [6,6,N]
    '''
    rho = np.asarray(rho, order='C')
    if xctype == 'GGA':
        order = 1
        nvar = 4
        fg = vxc[1].T
        frr = fxc[0].T
        frg = fxc[1].T
        fgg = fxc[2].T
    elif xctype == 'MGGA':
        order = 2
        nvar = 5
        fg = vxc[1].T
        frr, frg, fgg, ftt, frt, fgt = [fxc[i].T for i in [0, 1, 2, 4, 6, 9]]
    else:  # LDA
        order = 0
        nvar = 1
        frr = fxc[0].T

    ngrids = rho.shape[-1]
    if spin == 1:
        vp = np.empty((nvar, nvar, 2,2,ngrids))
        vp[0,0] = _stack_frr(frr)

        if order > 0:
            i3 = np.arange(3)
            qgg = _stack_fgg(fgg)
            qgg = np.einsum('abcdg,axg->xbcdg', qgg, rho[:,1:4])
            qgg = np.einsum('xbcdg,cyg->xybdg', qgg, rho[:,1:4])
            qgg[i3,i3] += _stack_fg(fg)
            vp[1:4,1:4] = qgg

            qrg = _stack_fg(frg.reshape(2,3,ngrids), axis=1)
            qrg = np.einsum('rabg,axg->xrbg', qrg, rho[:,1:4])
            vp[0,1:4] = qrg
            vp[1:4,0] = qrg.transpose(0,2,1,3)

        if order > 1:
            qgt = _stack_fg(fgt.reshape(3,2,ngrids), axis=0)
            qgt = np.einsum('abrg,axg->xbrg', qgt, rho[:,1:4])
            vp[1:4,4] = qgt
            vp[4,1:4] = qgt.transpose(0,2,1,3)

            qrt = frt.reshape(2,2,ngrids)
            vp[0,4] = qrt
            vp[4,0] = qrt.transpose(1,0,2)

            vp[4,4] = _stack_frr(ftt)

        vp = vp.transpose(2,0,3,1,4)

    else:
        vp = np.empty((nvar, nvar, ngrids))
        vp[0,0] = frr
        if order > 0:
            i3 = np.arange(3)
            qgg = 4 * fgg * rho[1:4] * rho[1:4,None]
            qgg[i3,i3] += fg * 2
            vp[1:4,1:4] = qgg
            vp[0,1:4] = vp[1:4,0] = 2 * frg * rho[1:4]
        if order > 1:
            vp[4,1:4] = vp[1:4,4] = 2 * fgt * rho[1:4]
            vp[0,4] = frt
            vp[4,0] = frt
            vp[4,4] = ftt
    return vp

def transform_kxc(rho, fxc, kxc, xctype, spin=0):
    r'''
    Transform libxc functional derivatives to the derivative tensor of
    parameters in rho:
        [[density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a],
         [density_b, (nabla_x)_b, (nabla_y)_b, (nabla_z)_b, tau_b]].
    The output tensor has the shape:
        * spin polarized
            LDA : [2,1,2,1,2,1,N]
            GGA : [2,4,2,4,2,4,N]
            MGGA: [2,6,2,6,2,6,N]
        * spin unpolarized
            LDA : [1,1,1,N]
            GGA : [4,4,4,N]
            MGGA: [6,6,6,N]
    '''
    rho = np.asarray(rho, order='C')
    if xctype == 'GGA':
        order = 1
        nvar = 4
        frg = fxc[1].T
        fgg = fxc[2].T
        frrr, frrg, frgg, fggg = [x.T for x in kxc[:4]]
    elif xctype == 'MGGA':
        order = 2
        nvar = 5
        frg = fxc[1].T
        fgg = fxc[2].T
        fgt = fxc[9].T
        frrr, frrg, frgg, fggg, frrt, frgt, frtt, fggt, fgtt, fttt = \
                [kxc[i].T for i in [0, 1, 2, 3, 5, 7, 10, 12, 15, 19]]
    else:  # LDA
        order = 0
        nvar = 1
        frrr = kxc[0].T

    ngrids = rho.shape[-1]
    if spin == 1:
        vp = np.empty((nvar, nvar, nvar, 2,2,2,ngrids))
        vp[0,0,0] = _stack_frrr(frrr)
        if order > 0:
            i3 = np.arange(3)
            qggg = _stack_fggg(fggg)
            qggg = np.einsum('abcdefg,axg->xbcdefg', qggg, rho[:,1:4])
            qggg = np.einsum('xbcdefg,cyg->xybdefg', qggg, rho[:,1:4])
            qggg = np.einsum('xybdefg,ezg->xyzbdfg', qggg, rho[:,1:4])
            qgg = _stack_fgg(fgg)
            qgg = np.einsum('abcdg,axg->xbcdg', qgg, rho[:,1:4])
            for i in range(3):
                qggg[:,i,i] += qgg
                qggg[i,:,i] += qgg.transpose(0,2,1,3,4)
                qggg[i,i,:] += qgg.transpose(0,2,3,1,4)
            vp[1:4,1:4,1:4] = qggg

            qrgg = _stack_fgg(frgg.reshape(2,6,ngrids), axis=1)
            qrgg = np.einsum('rabcdg,axg->xrbcdg', qrgg, rho[:,1:4])
            qrgg = np.einsum('xrbcdg,cyg->xyrbdg', qrgg, rho[:,1:4])
            qrg = _stack_fg(frg.reshape(2,3,ngrids), axis=1)
            qrgg[i3,i3] += qrg
            vp[0,1:4,1:4] = qrgg
            vp[1:4,0,1:4] = qrgg.transpose(0,1,3,2,4,5)
            vp[1:4,1:4,0] = qrgg.transpose(0,1,3,4,2,5)

            qrrg = _stack_fg(frrg.reshape(3,3,ngrids), axis=1)
            qrrg = _stack_frr(qrrg, axis=0)
            qrrg = np.einsum('rsabg,axg->xrsbg', qrrg, rho[:,1:4])
            vp[0,0,1:4] = qrrg
            vp[0,1:4,0] = qrrg.transpose(0,1,3,2,4)
            vp[1:4,0,0] = qrrg.transpose(0,3,1,2,4)

        if order > 1:
            qggt = _stack_fgg(fggt.reshape(6,2,ngrids), axis=0)
            qggt = np.einsum('abcdrg,axg->xbcdrg', qggt, rho[:,1:4])
            qggt = np.einsum('xbcdrg,cyg->xybdrg', qggt, rho[:,1:4])
            qgt = _stack_fg(fgt.reshape(3,2,ngrids), axis=0)
            qggt[i3,i3] += qgt
            vp[1:4,1:4,4] = qggt
            vp[1:4,4,1:4] = qggt.transpose(0,1,2,4,3,5)
            vp[4,1:4,1:4] = qggt.transpose(0,1,4,2,3,5)

            qgtt = _stack_frr(fgtt.reshape(3,3,ngrids), axis=1)
            qgtt = _stack_fg(qgtt, axis=0)
            qgtt = np.einsum('abrsg,axg->xbrsg', qgtt, rho[:,1:4])
            vp[1:4,4,4] = qgtt
            vp[4,1:4,4] = qgtt.transpose(0,2,1,3,4)
            vp[4,4,1:4] = qgtt.transpose(0,2,3,1,4)

            qrgt = _stack_fg(frgt.reshape(2,3,2,ngrids), axis=1)
            qrgt = np.einsum('rabsg,axg->xrbsg', qrgt, rho[:,1:4])
            vp[0,1:4,4] = qrgt
            vp[0,4,1:4] = qrgt.transpose(0,1,3,2,4)
            vp[1:4,0,4] = qrgt.transpose(0,2,1,3,4)
            vp[4,0,1:4] = qrgt.transpose(0,3,1,2,4)
            vp[1:4,4,0] = qrgt.transpose(0,2,3,1,4)
            vp[4,1:4,0] = qrgt.transpose(0,3,2,1,4)

            qrrt = _stack_frr(frrt.reshape(3,2,ngrids), axis=0)
            vp[0,0,4] = qrrt
            vp[0,4,0] = qrrt.transpose(0,2,1,3)
            vp[4,0,0] = qrrt.transpose(2,0,1,3)

            qrtt = _stack_frr(frtt.reshape(2,3,ngrids), axis=1)
            vp[0,4,4] = qrtt
            vp[4,0,4] = qrtt.transpose(1,0,2,3)
            vp[4,4,0] = qrtt.transpose(1,2,0,3)

            vp[4,4,4] = _stack_frrr(fttt, axis=0)

        vp = vp.transpose(3,0,4,1,5,2,6)

    else:
        vp = np.empty((nvar, nvar, nvar, ngrids))
        vp[0,0,0] = frrr
        if order > 0:
            i3 = np.arange(3)
            qggg = 8 * fggg * rho[1:4] * rho[1:4,None] * rho[1:4,None,None]
            qgg = 4 * fgg * rho[1:4]
            for i in range(3):
                qggg[i,i,:] += qgg
                qggg[i,:,i] += qgg
                qggg[:,i,i] += qgg
            vp[1:4,1:4,1:4] = qggg

            qrgg = 4 * frgg * rho[1:4] * rho[1:4,None]
            qrgg[i3,i3] += frg * 2
            vp[0,1:4,1:4] = qrgg
            vp[1:4,0,1:4] = qrgg
            vp[1:4,1:4,0] = qrgg

            qrrg = 2 * frrg * rho[1:4]
            vp[0,0,1:4] = qrrg
            vp[0,1:4,0] = qrrg
            vp[1:4,0,0] = qrrg

        if order > 1:
            qggt = 4 * fggt * rho[1:4] * rho[1:4,None]
            qggt[i3,i3] += fgt * 2
            vp[1:4,1:4,4] = qggt
            vp[1:4,4,1:4] = qggt
            vp[4,1:4,1:4] = qggt

            qgtt = 2 * fgtt * rho[1:4]
            vp[1:4,4,4] = qgtt
            vp[4,1:4,4] = qgtt
            vp[4,4,1:4] = qgtt

            qrgt = 2 * frgt * rho[1:4]
            vp[0,1:4,4] = qrgt
            vp[0,4,1:4] = qrgt
            vp[1:4,0,4] = qrgt
            vp[4,0,1:4] = qrgt
            vp[1:4,4,0] = qrgt
            vp[4,1:4,0] = qrgt

            vp[0,0,4] = frrt
            vp[0,4,0] = frrt
            vp[4,0,0] = frrt
            vp[0,4,4] = frtt
            vp[4,0,4] = frtt
            vp[4,4,0] = frtt
            vp[4,4,4] = fttt
    return vp

def transform_lxc(rho, fxc, kxc, lxc, xctype, spin=0):
    r'''
    Transform libxc vxc functional output to the derivative tensor of parameters
    in rho: [density, nabla_x, nabla_y, nabla_z, tau]. The output tensor has the
    shape:
        * spin polarized
            LDA : [2,1,2,1,2,1,2,1,N]
            GGA : [2,4,2,4,2,4,2,4,N]
            MGGA: [2,6,2,6,2,6,2,6,N]
        * spin unpolarized
            LDA : [1,1,1,1,N]
            GGA : [4,4,4,4,N]
            MGGA: [6,6,6,6,N]
    '''
    raise NotImplementedError

def _stack_fg(fg, axis=0):
    '''fg [uu, ud, dd] -> [[uu*2, ud], [du, dd*2]]'''
    qg = _stack_frr(fg, axis)
    if axis == 0:
        qg[0,0] *= 2
        qg[1,1] *= 2
    elif axis == 1:
        qg[:,0,0] *= 2
        qg[:,1,1] *= 2
    elif axis == 2:
        qg[:,:,0,0] *= 2
        qg[:,:,1,1] *= 2
    else:
        raise NotImplementedError
    return qg

def _stack_frr(frr, axis=0):
    '''frr [u_u, u_d, d_d] -> [[u_u, u_d], [d_u, d_d]]'''
    if frr.shape[axis] != 3:
        frr = frr.reshape(frr.shape[:axis] + (3, -1) + frr.shape[axis+1:])
    slices = [slice(None)] * frr.ndim
    slices[axis] = [[0,1],[1,2]]
    return frr[tuple(slices)]

def _stack_fgg(fgg, axis=0):
    '''
    fgg [uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd] ->
    [[uu_uu, ud_ud, ud_dd],
     [ud_uu, ud_ud, ud_dd],
     [dd_uu, dd_ud, dd_dd]] -> tensor with shape [2,2, 2,2, ...]
    '''
    if fgg.shape[axis] != 6:
        fgg = fgg.reshape(fgg.shape[:axis] + (6, -1) + fgg.shape[axis+1:])
    slices = [slice(None)] * fgg.ndim
    slices[axis] = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
    qgg = fgg[tuple(slices)]
    return _stack_fg(_stack_fg(qgg, axis=axis+1), axis=axis)

def _stack_frrr(frrr, axis=0):
    '''
    frrr [u_u_u, u_u_d, u_d_d, d_d_d]
    -> tensor with shape [2, 2, 2, ...]
    '''
    if frrr.shape[axis] != 4:
        frrr = frrr.reshape(frrr.shape[:axis] + (4, -1) + frrr.shape[axis+1:])
    slices = [slice(None)] * frrr.ndim
    slices[axis] = [[[0, 1], [1, 2]],
                    [[1, 2], [2, 3]]]
    return frrr[tuple(slices)]

def _stack_fggg(fggg, axis=0):
    '''
    fggg [uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, ud_ud_dd, ud_dd_dd, dd_dd_dd]
    -> tensor with shape [2,2, 2,2, 2,2, ...]
    '''
    if fggg.shape[axis] != 10:
        fggg = fggg.reshape(fggg.shape[:axis] + (10, 2) + fggg.shape[axis+1:])
    slices = [slice(None)] * fggg.ndim
    slices[axis] = [[[0, 1, 2], [1, 3, 4], [2, 4, 5]],
                    [[1, 3, 4], [3, 6, 7], [4, 7, 8]],
                    [[2, 4, 5], [4, 7, 8], [5, 8, 9]]]
    qggg = fggg[tuple(slices)]
    return _stack_fg(_stack_fg(_stack_fg(qggg, axis=axis+2), axis=axis+1), axis=axis)

def compress(vp, spin=0):
    if spin != 0:  # spin polarized
        shape = vp.shape
        comp_shape = [shape[i*2]*shape[i*2+1] for i in range(0, vp.ndim-1, 2)]
        vp = vp.reshape(comp_shape + [shape[-1]])

    order = vp.ndim - 1
    if order < 2:
        pass
    elif order == 2:  # 2nd derivatives
        vp = vp[np.tril_indices(vp.shape[0])]
    elif order == 3:  # 2nd derivatives
        nd = vp.shape[0]
        vp = np.vstack([vp[i][np.tril_indices(i+1)] for i in range(nd)])
    else:
        raise NotImplementedError('High order derivatives')
    return vp

def decompress(vp, spin=0):
    order = vp.ndim - 1
    ngrids = vp.shape[-1]
    if order < 2:
        out = vp
    elif order == 2:  # 2nd derivatives
        nd = vp.shape[0]
        out = np.empty((nd, nd, ngrids))
        idx, idy = np.tril_indices(nd)
        out[idx,idy] = vp
        out[idy,idx] = vp
    elif order == 3:  # 2nd derivatives
        nd = vp.shape[0]
        out = np.empty((nd, nd, nd, ngrids))
        p1 = 0
        for i in range(nd):
            idx, idy = np.tril_indices(i+1)
            p0, p1 = p1, p1+idx.size
            out[i,idx,idy] = vp[p0:p1]
            out[i,idy,idx] = vp[p0:p1]
            out[idx,i,idy] = vp[p0:p1]
            out[idy,i,idx] = vp[p0:p1]
            out[idx,idy,i] = vp[p0:p1]
            out[idy,idx,i] = vp[p0:p1]
    else:
        raise NotImplementedError('High order derivatives')

    if spin != 0:  # spin polarized
        shape = out.shape
        nvar = shape[0] // 2
        decomp_shape = [(2, nvar)] * order + [ngrids]
        # reshape to (2,n,2,n,...,ngrids)
        out = out.reshape(np.hstack(decomp_shape))
    return out

def ud2ts(v_ud):
    '''XC derivatives spin-up/spin-down representations to
    total-density/spin-density representations'''
    c = np.array([[.5,  .5],    # vrho = (va + vb) / 2
                  [.5, -.5]])   # vs   = (va - vb) / 2
    if v_ud.ndim == 3:  # vxc
        v_ts = np.einsum('ra,axg->rxg', c, v_ud)
    elif v_ud.ndim == 5:  # fxc
        v_ts = np.einsum('ra,axbyg->rxbyg', c, v_ud)
        v_ts = np.einsum('sb,rxbyg->rxsyg', c, v_ts)
    elif v_ud.ndim == 7:  # kxc
        v_ts = np.einsum('ra,axbyczg->rxbyczg', c, v_ud)
        v_ts = np.einsum('sb,rxbyczg->rxsyczg', c, v_ts)
        v_ts = np.einsum('tc,rxsyczg->rxsytzg', c, v_ts)
    else:
        raise NotImplementedError(f'Shape {v_ud.shape} not supported')
    return v_ts

def ts2ud(v_ts):
    '''XC derivatives total-density/spin-density representations to
    spin-up/spin-down representations'''
    c = np.array([[1,  1],    # va = vrho + vs
                  [1, -1]])   # vb = vrho - vs
    if v_ts.ndim == 3:  # vxc
        v_ud = np.einsum('ra,axg->rxg', c, v_ts)
    elif v_ts.ndim == 5:  # fxc
        v_ud = np.einsum('ra,axbyg->rxbyg', c, v_ts)
        v_ud = np.einsum('sb,rxbyg->rxsyg', c, v_ud)
    elif v_ts.ndim == 7:  # kxc
        v_ud = np.einsum('ra,axbyczg->rxbyczg', c, v_ts)
        v_ud = np.einsum('sb,rxbyczg->rxsyczg', c, v_ud)
        v_ud = np.einsum('tc,rxsyczg->rxsytzg', c, v_ud)
    else:
        raise NotImplementedError(f'Shape {v_ts.shape} not supported')
    return v_ud
################## xc_deriv ####################

if __name__ == '__main__':
    rho_tm = np.array([
        # density, nabla_x, nabla_y, nabla_z, tau  
        [0.65,     0.42,    0.48,    0.49,    0.10],  # total density
        [0.05,     0.22,   -0.02,    0.11,    0.01],  # mx
        [0.14,     0.01,   -0.32,    0.01,    0.01],  # my
        [0.12,    -0.12,   -0.01,    0.28,    0.01],  # mz
    ]).reshape(4,5,1)
    exc, vxc = eval_xc_eff(mcfun_eval_xc_wrapper('m06l', 'MGGA'), rho_tm, deriv=1, max_workers=4)[:2]
    print(exc)
    print(vxc)
