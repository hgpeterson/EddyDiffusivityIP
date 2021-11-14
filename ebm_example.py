import numpy as np
import matplotlib.pyplot as plt
from ebm import EBM

# plt.style.use("plots.mplstyle")

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
plt.plot(x, 1e4*ebm.D)
plt.xlabel("$x$")
plt.ylabel("$D$ (kg m$^{-2}$ s$^{-1}$)")
plt.tight_layout()
plt.savefig("D.png")
print("D.png")
plt.close()

# plot
plt.plot(x, h_tilde/1e3, label="model")
plt.plot(x, h/1e3, label="truth")
plt.xlabel("$x$")
plt.ylabel("$h$ (kJ kg$^{-1}$)")
plt.legend()
plt.tight_layout()
plt.savefig("h.png")
print("h.png")
plt.close()
