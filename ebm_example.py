import numpy as np
import matplotlib.pyplot as plt
from ebm import EBM

plt.style.use("plots.mplstyle")

# setup test problem
n = 2**8
x = np.linspace(-1, 1, n)
Q = 1365/np.pi*np.cos(np.arcsin(x))
Q -= np.trapz(Q, x)/2
D = 2.6e-4*np.ones(n)
hs = hn = 2.4e5

# solve
ebm = EBM(x, Q, D, hs, hn)
h = ebm.solve()

# plot
plt.plot(x, h/1e3)
plt.xlabel("$x$")
plt.ylabel("$h$ (kJ kg$^{-1}$)")
plt.tight_layout()
plt.savefig("h.png")
print("h.png")
plt.close()

# # real data
# data = np.load("Ts.npz")
# Ts = data["Ts"]
# lat = data["lat"]
# n = len(lat)

# h = h(Ts)

# x = np.sin(lat*np.pi/180)
# # Q = 1365/np.pi*np.cos(np.arcsin(x))
# # Q -= np.trapz(Q, x)/2
# # D = 0.649*np.ones(n) # W m-2 K-1
# # hs = Ts[0]
# # hn = Ts[-1]
# # ebm = EBM(x, Q, D)
# # h = ebm.solve()
# plt.plot(x, h/1e3)
# plt.show()
