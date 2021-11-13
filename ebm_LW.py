import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
import os

# Constants 
ps = 98000     # surface pressure (kg m-1 s-2)
cp = 1005      # specific heat capacity at constant pressure (J kg-1 K-1)
RH = 0.8       # relative humidity (0-1)
Lv = 2257000   # latent heat of vaporization (J kg-1)

def qsat(t, p):
    """
        qsat = qsat(t, p)

    Computes saturation specific humidity (qsat), given inputs temperature (t) in K and
    pressure (p) in hPa.
                            
    Buck (1981, J. Appl. Meteorol.)
    """
    tc=t-273.16;
    tice=-23;
    t0=0;
    Rd=287.04;
    Rv=461.5;
    epsilon=Rd/Rv;
    ewat=(1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*tc/(240.97+tc))
    eice=(1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*tc/(272.55+tc))
    eint=eice+(ewat-eice)*((tc-tice)/(t0-tice))**2
    esat=eint
    esat[np.where(tc<tice)]=eice[np.where(tc<tice)]
    esat[np.where(tc>t0)]=ewat[np.where(tc>t0)]
    qsat = epsilon*esat/(p-esat*(1-epsilon));
    return qsat

# datasets of T, q, and h to be able to convert from one to the other
T_dataset = np.arange(100, 400, 1e-3)
q_dataset = qsat(T_dataset, ps/100)
h_dataset = cp*T_dataset + RH*q_dataset*Lv

def h_from_T(T):
    """
        h = h_from_T(T)

    Compute h = cp*T + RH*Lv*q(T).
    """
    return cp*T + RH*Lv*qsat(T, ps/100)

def T_from_h(h):
    """
        T = T_from_h(h)

    Compute T from h = cp*T + RH*Lv*q(T).
    """
    return T_dataset[np.searchsorted(h_dataset, h)]

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
    and S is the SW energy input. The model uses the finite element method. It 
    is designed to take x, S, and D as inputs and output h.
    """

    def __init__(self, x, S, D, A, B):
        # givens
        self.x = x   # grid (sine latitude)
        self.S = S   # net energy input (W m-2)
        self.D = D   # eddy diffusivity (kg m-2 s-1)
        self.A = A
        self.B = B

        # initial MSE 
        hmax = 350000
        hmin = 250000
        self.h = 2/3*hmax + 1/3*hmin - 1/3*(hmax - hmin)*(3*self.x**2 - 1)

        # computables
        self.n = len(x) # number of grid points

    def _generate_linear_system(self):
        """
            A, b = self._generate_linear_system()

        Compute the matrix `A` and vector `b` that characterize the system
            A h = b
        we wish to solve. We use the stamping method.
        """
        rcd = np.empty([0, 3])    # row-column-data array: components of CSC matrix A
        b = np.zeros((self.n, 1)) # right-hand-side vector b

        # compute T
        self.T = T_from_h(self.h)

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
            T = lerp([x0, self.T[i]], [x1, self.T[i+1]])

            # integrate (1 - x^2)*D and Q*(mx + b) using Gaussian quadrature
            intD  = gquad2(lambda x: (1 - x**2)*D(x),  x0, x1)
            intS0 = gquad2(lambda x: (b0 + m0*x)*S(x), x0, x1)
            intS1 = gquad2(lambda x: (b1 + m1*x)*S(x), x0, x1)
            intT0 = gquad2(lambda x: (b0 + m0*x)*(self.A + self.B*T(x)), x0, x1)
            intT1 = gquad2(lambda x: (b1 + m1*x)*(self.A + self.B*T(x)), x0, x1)

            # stamp integral components for diffusivity term
            rcd = add_node(rcd, i, i,     m0*m0*intD)
            rcd = add_node(rcd, i, i+1,   m0*m1*intD)
            rcd = add_node(rcd, i+1, i,   m1*m0*intD)
            rcd = add_node(rcd, i+1, i+1, m1*m1*intD)

            # stamp integral components for RHS
            b[i]   += intS0
            b[i+1] += intS1

            # # stamp integral components for LW term
            # b[i]   -= intT0
            # b[i+1] -= intT1
            rcd = add_node(rcd, i, i,     gquad2(lambda x: self.B*(b0 + m0*x)*(b0 + m0*x), x0, x1))
            rcd = add_node(rcd, i, i+1,   gquad2(lambda x: self.B*(b0 + m0*x)*(b1 + m1*x), x0, x1))
            rcd = add_node(rcd, i+1, i,   gquad2(lambda x: self.B*(b1 + m1*x)*(b0 + m0*x), x0, x1))
            rcd = add_node(rcd, i+1, i+1, gquad2(lambda x: self.B*(b1 + m1*x)*(b1 + m1*x), x0, x1))

        # assemble sparse matrix
        A = csc_matrix((rcd[:, 2], (rcd[:, 0], rcd[:, 1])), shape=(self.n, self.n))

        # return as LU factorization
        return splu(A), b

    def solve(self):
        """
            h = self.solve()

        Generate and solve A h = b.
        """
        for i in range(10):
            # generate A and b
            A, b = self._generate_linear_system()

            # solve system
            h = A.solve(b) 

            # print error
            print(np.max(np.abs(h - self.h)/self.h))

            # set h
            self.h = h

        return h

# test problem
n = 2**8
x = np.linspace(-1, 1, n)
S = 1365/np.pi*np.cos(np.arcsin(x))*(1 - 0.33)
A = 203.3
S -= A
B = 2.09
D = 2.6e-4*np.ones(n)

ebm = EBM(x, S, D, A, B)
h = ebm.solve()

plt.plot(x, h/1e3)
plt.xlabel("$x$")
plt.ylabel("$h$ (kJ kg$^{-1}$)")
plt.tight_layout()
plt.savefig("h.png")
print("h.png")
plt.close()
