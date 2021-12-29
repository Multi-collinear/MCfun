import os
from setuptools import setup, find_packages

NAME            = 'MCfun'
MAINTAINERS     = 'Zhichen Pu(hoshishin), Hao Li(Haskiy)'
AUTHOR          = 'Yunlong Xiao, Zhichen Pu(hoshishin), Hao Li(Haskiy)'
DESCRIPTION     = 'A library to calculate multi-collinear functionl'
URL             = 'https://github.com/Multi-collinear/MCfun'

def get_version():
    topdir = os.path.abspath(os.path.join(__file__, '..'))
    with open(os.path.join(topdir, 'mcfun', '__init__.py'), 'r') as f:
        for line in f.readlines():
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise ValueError("Version string not found")
VERSION = get_version()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    #include *.so *.dat files. They are now placed in MANIFEST.in
    package_data={'': ['*.so', '*.dylib', '*.dll', '*.dat', '*.npy']},
    include_package_data=True,  # include everything in source control
    packages=find_packages(exclude=['*test*', '*examples*']),
    install_requires=['numpy', 'typing', 'nptyping'],
    #extras_require=EXTRAS,
)
