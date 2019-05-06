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

    statesReal = hfReal.get('dataset')[()]
    statesImag = hfImag.get('dataset')[()]

    states = statesReal + statesImag * 1j
    states = states.transpose()
    return states

def LoadData(file):
    """
    Load data from the C++ format as HDF5 in numpy.
    """
    hf = h5py.File(file, 'r')
    data = hf.get('dataset')[()]
    states = states.transpose()
    return states
