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
data = np.load("data/h_Q_CNRM-CM5.npz")
h = data["h"]
x = data["x"]
Q = data["Q"]
hs = h[0]
hn = h[-1]
D = 2.6e-4*np.ones(len(x))

# solve
ebm = EBM(x, Q, D, hs, hn)
h_tilde = ebm.solve()

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
