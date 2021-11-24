import numpy as np
import matplotlib.pyplot as plt
from ebm import EBM

plt.style.use("plots.mplstyle")

# generate synthetic problem data
n = 2**7
x = np.linspace(-1, 1, n)
Q = 1365/4*(1 - 0.3)*np.cos(np.arcsin(x))
Q -= np.trapz(Q, x)/2
hs = hn = 2.5e5
spectral = True
n_polys = 5
D = np.zeros(n_polys)
D[0] = 2.6e-4 
D[4] = -1e-4 

# solve
ebm = EBM(x, Q, D, hs, hn, spectral=spectral)
h = ebm.solve()

# save
np.savez("data/h_Q_synthetic.npz", h=h, Q=Q, x=x, D=ebm.D)
print("data/h_Q_synthetic.npz")

# plot Q
fig, ax = plt.subplots()
ax.plot(x, Q)
ax.set_xlabel(r"latitude $\phi$ (degrees)")
ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_ylabel(r"net energy input $Q$ (W m$^{-2}$)")
plt.savefig("Q.png")
print("Q.png")
plt.close()

# plot D
fig, ax = plt.subplots()
ax.plot(x, 1e4*ebm.D)
ax.set_xlabel(r"latitude $\phi$ (degrees)")
ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_ylabel(r"diffusivity $D$ ($\times 10^{-4}$ kg m$^{-2}$ s$^{-1}$)")
plt.savefig("D.png")
print("D.png")
plt.close()

# plot
fig, ax = plt.subplots()
ax.plot(x, h/1e3)
ax.set_xlabel(r"latitude $\phi$ (degrees)")
ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_ylabel("moist static energy $h$ (kJ kg$^{-1}$)")
plt.savefig("h.png")
print("h.png")
plt.close()