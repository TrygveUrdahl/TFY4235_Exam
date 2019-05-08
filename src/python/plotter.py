#!/usr/local/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import h5py
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.serif'] = 'CMU Serif'
mpl.rcParams['font.family'] = 'serif'

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
    fig, axes = plt.subplots()
    axes.set_xlabel("")
    axes.set_ylabel("")
    im = axes.matshow(data)
    #fig.colorbar(im)
    plt.axis("off")
    fig.tight_layout()
    #plt.savefig("./output/atomconfig08.pdf", bbox_inches="tight", pad=0)
    plt.show()

def plotEigvals(file):
    data = LoadData(file)
    fig, axes = plt.subplots()
    xaxis = [i for i in range(data.size)]
    axes.set_xlabel("")
    axes.set_ylabel("Eigenenergy")
    axes.plot(xaxis, data)
    fig.tight_layout()
    #plt.savefig("./output/eigenenergy.pdf", bbox_inches="tight", pad=0)
    plt.show()

def plotEnthalpys(file):
    fig, axes = plt.subplots()
    data = LoadData(file)
    xaxis = np.linspace(0,1,data.size)
    axes.set_xlabel("$x_A$")
    axes.set_ylabel("$\Delta F(a,x_A) [eV]$")
    axes.plot(xaxis, data)
    fig.tight_layout()
    plt.savefig("./output/enthalpysw09.pdf", bbox_inches="tight", pad=0)
    plt.show()

def plotIterationCount(file):
    fig, axes = plt.subplots()
    data = LoadData(file)
    xaxis = np.linspace(0,1,data.size)
    axes.set_xlabel("$x_A$")
    axes.set_ylabel("Iterations")
    axes.plot(xaxis, data)
    fig.tight_layout()
    plt.savefig("./output/iterationcountAve.pdf", bbox_inches="tight", pad=0)
    plt.show()

def findPotentials():
    beta = 10
    r = np.linspace(0.7,1.5,100)
    Vaa = 5.0 * ((0.8/r)**6 - np.exp(-r/1.95))
    Vbb = 6.0 * ((0.7/r)**6 - np.exp(-r/0.70))
    Vab = 4.5 * ((0.85/r)**6 - np.exp(-r/1.90))

    fig, axes = plt.subplots()
    axes.plot(r, Vaa, label="$V_{AA}$")
    axes.plot(r, Vab, label="$V_{AB}$")
    axes.plot(r, Vbb, label="$V_{BB}$")


    axes.axvline(0.9, linestyle="--", color="black", alpha=0.5)
    axes.axvline(1.3, linestyle="--", color="black", alpha=0.5)
    #axes.axhline(0.0, linestyle="--", color="black", alpha=0.5)

    axes.set_xlabel("Atom distance $r$ [Å]")
    axes.set_ylabel("Potential $V_{n_1 n_2}$ [eV]")
    plt.legend()
    fig.tight_layout()
    plt.savefig("./output/potentials.pdf", bbox_inches="tight", pad=0)
    plt.show()

plotAtomConfig("./output/config.h5")
#plotEigvals("./output/eigvals.h5")
#plotEnthalpys("./output/enthalpys.h5")
#plotIterationCount("./output/iterationCount.h5")
#findPotentials()
