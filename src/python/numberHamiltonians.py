#!/usr/local/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.serif'] = 'CMU Serif'
mpl.rcParams['font.family'] = 'serif'

def logNFac(n):
    return sum([math.log(i + 1) for i in range(n)])

N = 10
M = 20
numParticles = N * M
numA = range(numParticles)
numB = [numParticles - a for a in numA]
logNumHamiltonians = [logNFac(numParticles) - logNFac(numA[i]) - logNFac(numB[i]) for i in range(numParticles)]
xaxis = np.linspace(0,1,numParticles)


fig, axes = plt.subplots()
plt.plot(xaxis, logNumHamiltonians)
axes.set_xlabel("$x_A$")
axes.set_ylabel("$log(n_{Hamiltonians})$")
fig.tight_layout()
plt.savefig("./output/numhamiltonians.pdf")
plt.show()
