import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.special import legendre
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt
import os
plt.style.use("plots.mplstyle")

# Constants 
cp = 1e3    # specific heat capacity at constant pressure (J kg-1 K-1)
RH = 0.8    # relative humidity (0-1)
Lv = 2.26e6 # latent heat of vaporization (J kg-1)

def add_node(rcd, r, c, d):
    """
        rcd = add_node(rcd, r, c, d)

    Add a node to a CSC sparse matrix defined by `rcd`. The new node is of the form
        row = `r`
        column = `c`
        data = `d`
    and `rcd` is an N x 3 array of rows, columns, and data.
    """
    return np.append(rcd, [[r, c, d]], axis=0)

def basis_vector(I, i):
    """
        m, b = basis_vector(I, i)

    Compute the slope `m` and intercept `b` of a basis vector phi defined on
    the interval I = [x0, x1] such that phi(xi) = 1 and phi(xj) = 0 where
    j = (i + 1) % 2.
    """
    j = (i + 1) % 2
    m = -1/(I[j] - I[i])
    b = I[j]/(I[j] - I[i])
    return m, b 

def gquad2(f, a, b):
    """
        int = gquad(f, a, b)

    Compute integral of f(x) on interval [a, b] using Gaussian quadrature.
    """
    x = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    x = (b - a)/2 * x + (a + b)/2
    return (b - a)/2 * (f(x[0]) + f(x[1]))

def lerp(p0, p1):
    """
        f(x) = lerp(p0, p1)

    Linearly interpolate between two points p0 = [x0, y0] and p1 = [x1, y1].
    """
    return lambda x: p0[1]*(x - p1[0])/(p0[0] - p1[0]) + p1[1]*(x - p0[0])/(p1[0] - p0[0])

class EBM():
    """
    Energy Balance Model 

    This model solves the ODE
        -d/dx [ D(x) (1 - x**2) d/dx h(x) ] = S - (A + B*T)
    where D is the eddy diffusivity, x = sine latitude, h is moist static energy,
    S is the SW energy input, and A and B linearize the LW energy output. The model 
    uses the finite element method. It is designed to take x, S, D, A, and B as inputs 
    and output h.
    """

    def __init__(self, x, S, D, A, B, spectral=False):
        # params
        self.x = x  # grid (sine latitude)
        self.S = S  # SW energy input (W m-2)
        self.A = A  # constant term from linearized LW input (W m-2) 
        self.B = B  # linear term from linearized LW input (W m-2 K-1)
        self.n = len(x) # number of grid points

        # linearize T(h) = a + b*h
        T0 = 273.15 # triple point (K)
        q0 = 3.9e-3 # sat spec hum q at T0 and ps
        beta = 0.07 # dq/dT at T0 and ps
        self.a = -RH*Lv*q0*(1 - beta*T0) / (cp + RH*Lv*q0*beta)
        self.b = 1 / (cp + RH*Lv*q0*beta)

        # compute eddy diffusivity (kg m-2 s-1)
        if spectral:
            # # using Legendre polynomials
            # legendre_polys = np.zeros((self.n, len(D)))
            # for i in range(len(D)):
            #     legendre_polys[:, i] = legendre(i)(x)
            # self.D = np.dot(legendre_polys, D)

            # gaussians at -90:30:90
            x0 = np.sin(np.pi/180 * np.arange(-90, 91, 30))
            self.D = np.zeros(len(x))
            for i in range(len(x0)):
                self.D += D[i]*np.exp(-(x - x0[i])**2/(2*0.5**2))

            # # just interpolating
            # cs = CubicSpline(x[1::self.n//(len(D) - 1)], D)
            # self.D = cs(x) 

            # plt.plot(x, self.D)
            # plt.savefig("D.png")
            # os.sys.exit()
        else:
            # given directly
            self.D = D   

    def _generate_linear_system(self):
        """
            A, b = self._generate_linear_system()

        Compute the matrix `A` and vector `b` that characterize the system
            A h = b
        we wish to solve. We use the stamping method.
        """
        rcd = np.empty([0, 3])    # row-column-data array: components of CSC matrix A
        b = np.zeros((self.n, 1)) # right-hand-side vector b

        # assemble A and b using stamping
        for i in range(self.n-1):
            # interval I = [x0, x1]
            x0 = self.x[i]
            x1 = self.x[i+1]

            # basis vectors on I
            m0, b0 = basis_vector([x0, x1], 0)
            m1, b1 = basis_vector([x0, x1], 1)

            # approximate D, T, and S as linear on I
            D = lerp([x0, self.D[i]], [x1, self.D[i+1]])
            S = lerp([x0, self.S[i]], [x1, self.S[i+1]])

            # stamp integral components for diffusivity term: (1 - x**2)*D
            intD  = gquad2(lambda x: (1 - x**2)*D(x),  x0, x1)
            rcd = add_node(rcd, i, i,     m0*m0*intD)
            rcd = add_node(rcd, i, i+1,   m0*m1*intD)
            rcd = add_node(rcd, i+1, i,   m1*m0*intD)
            rcd = add_node(rcd, i+1, i+1, m1*m1*intD)

            # stamp integral components for LW term: A + B*T => A + a*B + b*B*h
            LW00 = self.b*self.B*gquad2(lambda x: (b0 + m0*x)*(b0 + m0*x), x0, x1)
            LW01 = self.b*self.B*gquad2(lambda x: (b0 + m0*x)*(b1 + m1*x), x0, x1)
            LW10 = self.b*self.B*gquad2(lambda x: (b1 + m1*x)*(b0 + m0*x), x0, x1)
            LW11 = self.b*self.B*gquad2(lambda x: (b1 + m1*x)*(b1 + m1*x), x0, x1)
            rcd = add_node(rcd, i, i,     LW00)
            rcd = add_node(rcd, i, i+1,   LW01)
            rcd = add_node(rcd, i+1, i,   LW10)
            rcd = add_node(rcd, i+1, i+1, LW11)
            intA0 = gquad2(lambda x: (b0 + m0*x)*(self.A + self.a*self.B), x0, x1)
            intA1 = gquad2(lambda x: (b1 + m1*x)*(self.A + self.a*self.B), x0, x1)
            b[i]   -= intA0
            b[i+1] -= intA1

            # stamp integral components for SW term: S
            intS0 = gquad2(lambda x: (b0 + m0*x)*S(x), x0, x1)
            intS1 = gquad2(lambda x: (b1 + m1*x)*S(x), x0, x1)
            b[i]   += intS0
            b[i+1] += intS1

        # assemble sparse matrix
        A = csc_matrix((rcd[:, 2], (rcd[:, 0], rcd[:, 1])), shape=(self.n, self.n))

        # return as LU factorization
        return splu(A), b

    def solve(self):
        """
            h = self.solve()

        Generate and solve A h = b.
        """
        A, b = self._generate_linear_system()
        h = A.solve(b) 

        return np.squeeze(h)

# # test problem
# n = 2**7
# x = np.linspace(-1, 1, n)
# S = 1365/4*(1 - 0.3)*np.cos(np.arcsin(x))
# B = 2.09 
# A = 203.3 - 273.15*B

# # D = 2.6e-4*np.ones(n)
# # ebm = EBM(x, S, D, A, B, spectral=False)

# # D = np.zeros(20)
# # D[0] = 2.6e-4
# # D[4] = -1e-4

# d = np.load("out.npz")
# D = d["us"][-1, :]

# ebm = EBM(x, S, D, A, B, spectral=True)
# h = ebm.solve()

# plt.plot(x, h/1e3)
# plt.xlabel("$x$")
# plt.ylabel("$h$ (kJ kg$^{-1}$)")
# plt.savefig("h.png")
# print("h.png")
# plt.close()