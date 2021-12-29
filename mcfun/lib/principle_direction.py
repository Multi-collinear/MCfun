#!/usr/bin/env python
r'''
Author: Pu Zhichen
Date: 2021-10-20 14:28:37
LastEditTime: 2021-11-04 14:11:08
LastEditors: Pu Zhichen
Description: 
FilePath: \MCfun\mcfun\lib\more_data.py

 May the force be with you!
'''

import numpy
import os

path = os.path.abspath(os.path.dirname(__file__))


# ~ Lebedev grid
N14   = numpy.load(path+'/D14.npy')
N38   = numpy.load(path+'/D38.npy')
N110   = numpy.load(path+'/D110.npy')
N266   = numpy.load(path+'/D266.npy')
N434   = numpy.load(path+'/D434.npy')
N770   = numpy.load(path+'/D770.npy')
N1454   = numpy.load(path+'/D1454.npy')
N3074   = numpy.load(path+'/D3074.npy')
N5810   = numpy.load(path+'/D5810.npy')

W14   = numpy.load(path+'/W14.npy')
W38   = numpy.load(path+'/W38.npy')
W110   = numpy.load(path+'/W110.npy')
W266   = numpy.load(path+'/W266.npy')
W434   = numpy.load(path+'/W434.npy')
W770   = numpy.load(path+'/W770.npy')
W1454   = numpy.load(path+'/W1454.npy')
W3074   = numpy.load(path+'/W3074.npy')
W5810   = numpy.load(path+'/W5810.npy')


# ~ Legendre grid
N20   = numpy.load(path+'/D20.npy')
N50   = numpy.load(path+'/D50.npy')
N100   = numpy.load(path+'/D100.npy')
N200   = numpy.load(path+'/D200.npy')
N800   = numpy.load(path+'/D800.npy')
N1800   = numpy.load(path+'/D1800.npy')
N5000   = numpy.load(path+'/D5000.npy')
N12800  = numpy.load(path+'/D12800.npy')

W20   = numpy.load(path+'/W20.npy')/numpy.pi*0.25
W50   = numpy.load(path+'/W50.npy')/numpy.pi*0.25
W100   = numpy.load(path+'/W100.npy')/numpy.pi*0.25
W200   = numpy.load(path+'/W200.npy')/numpy.pi*0.25
W800   = numpy.load(path+'/W800.npy')/numpy.pi*0.25
W1800   = numpy.load(path+'/W1800.npy')/numpy.pi*0.25
W5000   = numpy.load(path+'/W5000.npy')/numpy.pi*0.25
W12800  = numpy.load(path+'/W12800.npy')/numpy.pi*0.25


# ~ Fibonacci grid
N10   = numpy.load(path+'/D10.npy')
N40   = numpy.load(path+'/D40.npy')
N150   = numpy.load(path+'/D150.npy')
N300   = numpy.load(path+'/D300.npy')
N600   = numpy.load(path+'/D600.npy')
N1200   = numpy.load(path+'/D1200.npy')
N2400   = numpy.load(path+'/D2400.npy')
N4800   = numpy.load(path+'/D4800.npy')
N10000   = numpy.load(path+'/D10000.npy')
N50000   = numpy.load(path+'/D50000.npy')
N100000   = numpy.load(path+'/D100000.npy')

W10   = numpy.load(path+'/W10.npy')
W40   = numpy.load(path+'/W40.npy')
W150   = numpy.load(path+'/W150.npy')
W300   = numpy.load(path+'/W300.npy')
W600   = numpy.load(path+'/W600.npy')
W1200   = numpy.load(path+'/W1200.npy')
W2400   = numpy.load(path+'/W2400.npy')
W4800   = numpy.load(path+'/W4800.npy')
W10000   = numpy.load(path+'/W10000.npy')
W50000   = numpy.load(path+'/W50000.npy')
W100000   = numpy.load(path+'/W100000.npy')


NX = {
    # ~ Lebedev
    14    : N14   ,
    38    : N38   ,
    110   : N110  ,
    266   : N266  ,
    434   : N434  ,
    770   : N770  ,
    1454  : N1454 ,
    3074  : N3074 ,
    5810  : N5810 ,
    
    # ~ Legendre
    20    : N20   ,
    50    : N50   ,
    100   : N100  ,
    200   : N200  ,
    800   : N800  ,
    1800  : N1800 ,
    5000  : N5000 ,
    12800 : N12800,
    
    # ~ Fibonacci
    10    : N10    ,
    40    : N40    ,
    150   : N150   ,
    300   : N300   ,
    600   : N600   ,
    1200  : N1200  ,
    2400  : N2400  ,
    4800  : N4800  ,
    10000 : N10000 ,
    50000 : N50000 ,
    100000 : N100000,
}

WX = {
    # ~ Lebedev
    14    : W14   ,
    38    : W38   ,
    110   : W110  ,
    266   : W266  ,
    434   : W434  ,
    770   : W770  ,
    1454  : W1454 ,
    3074  : W3074 ,
    5810  : W5810 ,
    
    # ~ Legendre
    20    : W20   ,
    50    : W50   ,
    100   : W100  ,
    200   : W200  ,
    800   : W800  ,
    1800  : W1800 ,
    5000  : W5000 ,
    5810  : W5810 ,
    12800 : W12800,
    
    # ~ Fibonacci
    10    : W10    ,
    40    : W40    ,
    150   : W150   ,
    300   : W300   ,
    600   : W600   ,
    1200  : W1200  ,
    2400  : W2400  ,
    4800  : W4800  ,
    10000 : W10000 ,
    50000 : W50000 ,
    100000 : W100000
}




