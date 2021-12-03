# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:49:52 2021

@author: ronak
"""
from sys import modules
import glob
import numpy as np
import matplotlib.pyplot as plt
from ebm import EBM

np.random.seed(64)

plt.style.use("plots.mplstyle")

fig, ax = plt.subplots()
for name in glob.glob(r'out_*'):
    model = name.lstrip("out_h_Q_").rstrip(".npz")
    d= np.load(name)
    N = d["N"]
    x = d["x"]
    h = d["h"]
    us = d["us"]
    h_tildes = d["h_tildes"]
    Ds = d["Ds"]
    m = d["m"]
    C_inv_12 = d["C_inv_12"]
    gamma_inv_12 = d["gamma_inv_12"]

    ax.plot(x, 1e4*Ds[-1, :], label=str(model))

ax.legend(ncol=3, fontsize="x-small")
ax.set_xlabel(r"latitude $\phi$ (degrees)")
ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
plt.ylabel(r"diffusivity $D$ ($\times 10^{-4}$ kg m$^{-2}$ s$^{-1}$)")
plt.savefig('D_all.png')
