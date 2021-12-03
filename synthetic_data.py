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

# # save
# np.savez("data/h_Q_synthetic.npz", h=h, Q=Q, x=x, D=ebm.D)
# print("data/h_Q_synthetic.npz")

fig, ax = plt.subplots(1, 3, figsize=(6.5, 6.5/1.62/3))

ax[0].plot(x, Q)
ax[0].set_xlabel(r"latitude $\phi$ (degrees)")
ax[0].set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax[0].set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax[0].set_ylabel("$Q$ (W m$^{-2}$)")
ax[0].annotate("(a) net energy input", (-0.05, 1.05), xycoords="axes fraction")

ax[1].plot(x, 1e4*ebm.D)
ax[1].set_xlabel(r"latitude $\phi$ (degrees)")
ax[1].set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax[1].set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax[1].set_ylabel(r"$D$ ($\times 10^{-4}$ kg m$^{-2}$ s$^{-1}$)")
ax[1].annotate("(b) diffusivity", (-0.05, 1.05), xycoords="axes fraction")

ax[2].plot(x, h/1e3)
ax[2].set_xlabel(r"latitude $\phi$ (degrees)")
ax[2].set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax[2].set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax[2].set_ylabel("$h$ (kJ kg$^{-1}$)")
ax[2].annotate("(c) moist static energy", (-0.05, 1.05), xycoords="axes fraction")

plt.subplots_adjust(wspace=0.4)

plt.savefig("synth.pdf")
print("synth.pdf")
plt.close()