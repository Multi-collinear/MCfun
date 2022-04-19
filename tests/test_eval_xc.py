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
    rho_tm = np.array([
        [0.65, 0.42,  0.48, 0.12, 0.19],
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
    assert abs(exc*rho_tm[0,0] - -1.5989734810079324).max() < 1e-9
    assert abs(lib.fp(vxc) - -53.23651212325073).max() < 1e-6
    assert abs(lib.fp(fxc) - 2698.947349479585).max() < 1e-4


def test_eval_xc_eff_polarized_spins():
    rho_tm = np.array([
        [0.65, 0.42,  0.48, 0.12, 0.19],
        [0.45, 0.22, -0.02, 0.01, 0.11],
        [0.24, 0.01, -0.32,-0.01, 0.01],
        [0.12, 0.12, -0.01, 0.08, 0.28],
    ]).reshape(4,5,1)
    # Make strongly polarized
    rho_tm[1:] *= -rho_tm[0,0,0] / np.linalg.norm(rho_tm[1:,0,0])

    deriv = 2

    ni = numint2c.NumInt2C()
    ni.collinear = 'mc'
    fn = functools.partial(mcfun_fn_eval_xc, ni, 'slater')
    exc, vxc, fxc = eval_xc_eff(fn, rho_tm[:,0], deriv, spin_samples=50,
                                collinear_threshold=0.99, workers=1)
    assert abs(exc*rho_tm[0,0] - -0.5239375578786232).max() < 1e-9
    assert abs(lib.fp(vxc) - -0.5122824511618321).max() < 1e-6
    assert abs(lib.fp(fxc) - 1.120293450378787).max() < 1e-6

    fn = functools.partial(mcfun_fn_eval_xc, ni, 'pbe,')
    exc, vxc, fxc = eval_xc_eff(fn, rho_tm[:,:4], deriv, spin_samples=50,
                                collinear_threshold=0.99, workers=1)
    assert abs(exc*rho_tm[0,0] - -0.5259488616121332).max() < 1e-9
    assert abs(lib.fp(vxc) - -0.7659161259968917).max() < 1e-6
    assert abs(lib.fp(fxc) - 0.9654817249658881).max() < 1e-6

    fn = functools.partial(mcfun_fn_eval_xc, ni, 'm06l,')
    exc, vxc, fxc = eval_xc_eff(fn, rho_tm, deriv, spin_samples=50,
                                collinear_threshold=0.99, workers=1)
    assert abs(exc*rho_tm[0,0] - -0.5623667079838119).max() < 1e-9
    assert abs(lib.fp(vxc) - -0.2915344473027743).max() < 1e-6
    assert abs(lib.fp(fxc) - -1741313235.5147088).max() < 1e-2
