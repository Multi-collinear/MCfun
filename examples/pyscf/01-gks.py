import numpy as np
from pyscf import gto
mol = gto.M(atom='''
C    -0.00000000    0.00000000   -0.00000000 ;
H     0.00000000    1.0767   -0.00000000 ;
H     0.932449552254705   -0.53835   -0.00000000 ;
H    -0.932449552254705   -0.53835   -0.00000000 ;
            ''',
spin=1,
basis='6-31g',
verbose=4,
)
mf = mol.GKS(xc='pbe', collinear='mcol').run()
