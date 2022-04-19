import numpy as np
from mcfun import eval_xc_eff
from pyscf import lib
from pyscf.dft import numint2c

def test_eval_xc_eff():
    rho_tm = np.array([
        [0.65, 0.42,  0.48, 0.12, 0.19],
        [0.05, 0.22, -0.02, 0.01, 0.11],
        [0.14, 0.01, -0.32,-0.01, 0.01],
        [0.12, 0.12, -0.01, 0.08, 0.28],
    ]).reshape(4,5,1)
    deriv = 2

    ni = numint2c.NumInt2C()
    fn = ni.mcfun_eval_xc_wrapper('lda,')
    exc, vxc, fxc = eval_xc_eff(fn, rho_tm[:,0], deriv, spin_samples=50, workers=1)
    assert abs(exc - -0.6492576011409112).max() < 1e-9
    assert abs(lib.fp(vxc) - -0.7778270430728434).max() < 1e-6
    assert abs(lib.fp(fxc) - 1.8351022464343585).max() < 1e-6

    fn = ni.mcfun_eval_xc_wrapper('pbe,')
    exc, vxc, fxc = eval_xc_eff(fn, rho_tm[:,:4], deriv, spin_samples=50, workers=1)
    assert abs(exc - -0.65655765978509).max() < 1e-9
    assert abs(lib.fp(vxc) - -0.8584499366538568).max() < 1e-6
    assert abs(lib.fp(fxc) - 1.7759355503758736).max() < 1e-6

    fn = ni.mcfun_eval_xc_wrapper('m06l,')
    exc, vxc, fxc = eval_xc_eff(fn, rho_tm, deriv, spin_samples=50, workers=1)
    assert abs(exc - -1.8910661328309755).max() < 1e-9
    assert abs(lib.fp(vxc) - -53.23651212325073).max() < 1e-6
    assert abs(lib.fp(fxc) - 3055.720005714985).max() < 1e-4


test_eval_xc_eff()
