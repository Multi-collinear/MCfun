import functools
import numpy as np
from mcfun import eval_xc_eff
from pyscf import lib
from pyscf.dft import numint2c, xc_deriv

# * Converts the output of eval_xc_eff to the derivaties to total-density and
#   spin-density.
# * Makes it a global function than a closure so as to be callable by
#   multiprocessing
def mcfun_fn_eval_xc(ni, xc_code, rho, deriv):
    xctype = ni._xc_type(xc_code)
    exc, vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)
    if deriv > 0:
        vxc = xc_deriv.ud2ts(vxc)
    if deriv > 1:
        fxc = xc_deriv.ud2ts(fxc)
    if deriv > 2:
        kxc = xc_deriv.ud2ts(kxc)
    return exc, vxc, fxc, kxc

def test_eval_xc_eff():
    # Keep tau > |u| so all projected spin-channel kinetic densities are positive.
    rho_tm = np.array([
        [0.65, 0.42,  0.48, 0.12, 1.00],
        [0.05, 0.22, -0.02, 0.01, 0.11],
        [0.14, 0.01, -0.32,-0.01, 0.01],
        [0.12, 0.12, -0.01, 0.08, 0.28],
    ]).reshape(4,5,1)
    deriv = 2

    ni = numint2c.NumInt2C()
    ni.collinear = 'mc'
    fn = functools.partial(mcfun_fn_eval_xc, ni, 'slater')
    exc, vxc, fxc = eval_xc_eff(fn, rho_tm[:,0], deriv, spin_samples=50, workers=1)
    assert abs(exc*rho_tm[0,0] - -0.42389868683256454).max() < 1e-9
    assert abs(lib.fp(vxc) - -0.7778270430728434).max() < 1e-6
    assert abs(lib.fp(fxc) - 0.08187031004637776).max() < 1e-6

    fn = functools.partial(mcfun_fn_eval_xc, ni, 'pbe,')
    exc, vxc, fxc = eval_xc_eff(fn, rho_tm[:,:4], deriv, spin_samples=50, workers=1)
    assert abs(exc*rho_tm[0,0] - -0.4291423568933654).max() < 1e-9
    assert abs(lib.fp(vxc) - -0.8584499366538568).max() < 1e-6
    assert abs(lib.fp(fxc) - 0.0267922870678802).max() < 1e-6

    fn = functools.partial(mcfun_fn_eval_xc, ni, 'm06l,')
    exc, vxc, fxc = eval_xc_eff(fn, rho_tm, deriv, spin_samples=50, workers=1)
    assert abs(exc*rho_tm[0,0] - -0.45417783292400987).max() < 1e-9
    assert abs(lib.fp(vxc) - -0.9149340225469204).max() < 1e-6
    assert abs(lib.fp(fxc) - 0.5162336575016337).max() < 1e-4

    fn = functools.partial(mcfun_fn_eval_xc, ni, ',vwn')
    exc, vxc, fxc = eval_xc_eff(fn, rho_tm[:,0], deriv, spin_samples=50, workers=1)
    assert abs(exc*rho_tm[0,0] - -0.042954292392697185).max() < 1e-9
    assert abs(lib.fp(vxc) - -0.08852664943620511).max() < 1e-6
    assert abs(lib.fp(fxc) - -0.09119769357020457).max() < 1e-6

    fn = functools.partial(mcfun_fn_eval_xc, ni, ',lyp')
    exc, vxc, fxc = eval_xc_eff(fn, rho_tm[:,:4], deriv, spin_samples=50, workers=1)
    assert abs(exc*rho_tm[0,0] - -0.027322061029788333).max() < 1e-9
    assert abs(lib.fp(vxc) - -0.05010644359426393).max() < 1e-6
    assert abs(lib.fp(fxc) - -0.08607017853054048).max() < 1e-6

    fn = functools.partial(mcfun_fn_eval_xc, ni, ',tpss')
    exc, vxc, fxc = eval_xc_eff(fn, rho_tm, deriv, spin_samples=50, workers=1)
    assert abs(exc*rho_tm[0,0] - -0.03976671066210757).max() < 1e-9
    assert abs(lib.fp(vxc) - -0.09718877085448392).max() < 1e-6
    assert abs(lib.fp(fxc) - -0.013446799361438354).max() < 1e-6
