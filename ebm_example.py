import numpy as np
import matplotlib.pyplot as plt
from ebm import EBM

plt.style.use("plots.mplstyle")

# # test problem
# n = 2**8
# x = np.linspace(-1, 1, n)
# Q = 1365/np.pi*np.cos(np.arcsin(x))
# Q -= np.trapz(Q, x)/2
# D = 2.6e-4*np.ones(n)
# hs = hn = 2.4e5

# real data
data = np.load("data/h_Q_GFDL-CM3.npz")
h = data["h"]
x = data["x"]
Q = data["Q"]
hs = h[0]
hn = h[-1]
# spectral = False
# D = 2.6e-4*np.ones(len(x))
spectral = True
n_polys = 20
D = np.zeros(n_polys)
D[0] = 2.6e-4 
D[4] = -1e-4 

# solve
ebm = EBM(x, Q, D, hs, hn, spectral=spectral)
h_tilde = ebm.solve()

# plot D
fig, ax = plt.subplots()
ax.plot(x, 1e4*ebm.D)
ax.set_xlabel(r"latitude $\phi$ (degrees)")
ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_ylabel(r"$D$ ($\times 10^{-4}$ kg m$^{-2}$ s$^{-1}$)")
plt.tight_layout()
plt.savefig("D.png")
print("D.png")
plt.close()

# plot
fig, ax = plt.subplots()
ax.plot(x, h_tilde/1e3, label="model")
ax.plot(x, h/1e3, label="truth")
ax.set_xlabel(r"latitude $\phi$ (degrees)")
ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_ylabel("$h$ (kJ kg$^{-1}$)")
ax.legend()
plt.tight_layout()
plt.savefig("h.png")
print("h.png")
plt.close()
