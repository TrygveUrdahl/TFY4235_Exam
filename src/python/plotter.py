#!/usr/local/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import h5py
import pandas as pd
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'CMU Serif Roman 2'

def LoadComplexData(fileReal, fileImag):
    """
    Load complex data from the C++ format as HDF5 in numpy.
    """
    hfReal = h5py.File(fileReal, 'r')
    hfImag = h5py.File(fileImag, 'r')

    dataReal = hfReal.get('dataset')[()]
    dataImag = hfImag.get('dataset')[()]

    data = dataReal + dataImag * 1j
    data = data.transpose()
    return data

def LoadData(file):
    """
    Load data from the C++ format as HDF5 in numpy.
    """
    hf = h5py.File(file, 'r')
    data = hf.get('dataset')[()]
    data = data.transpose()
    return data

def plotAtomConfig(file):
    data = LoadData(file)
    plt.matshow(data)
    plt.show()

def plotEnthalpys(file):
    data = LoadData(file)
    xaxis = np.linspace(0,1,data.size)
    plt.plot(xaxis, data)
    plt.show()


# plotAtomConfig("./output/config.h5")
plotEnthalpys("./output/enthalpys.h5")
