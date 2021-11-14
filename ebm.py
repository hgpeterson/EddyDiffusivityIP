import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.special import legendre
# import matplotlib.pyplot as plt
# import os

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
        -d/dx [ D(x) (1 - x**2) d/dx h(x) ] = Q
    where D is the eddy diffusivity, x = sine latitude, h is moist static energy,
    and Q is the net energy input. The model uses the finite element method. It 
    is designed to take x, Q, and D as inputs and output h.
    """

    def __init__(self, x, Q, D, hs, hn, spectral=False):
        # load data
        self.x = x   # grid (sine latitude)
        self.Q = Q   # net energy input (W m-2)
        self.hs = hs # south pole moist static energy (J kg-1)
        self.hn = hn # north pole moist static energy (J kg-1)
        self.n = len(x) # number of grid points

        # compute eddy diffusivity (kg m-2 s-1)
        if spectral:
            # using Legendre polynomials
            legendre_polys = np.zeros((self.n, len(D)))
            for i in range(len(D)):
                legendre_polys[:, i] = legendre(i)(x)
            self.D = np.dot(legendre_polys, D)
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

            # approximate D and Q as linear on I
            D = lerp([x0, self.D[i]], [x1, self.D[i+1]])
            Q = lerp([x0, self.Q[i]], [x1, self.Q[i+1]])

            # integrate (1 - x^2)*D and Q*(mx + b) using Gaussian quadrature
            intD  = gquad2(lambda x: (1 - x**2)*D(x),  x0, x1)
            intQ0 = gquad2(lambda x: (b0 + m0*x)*Q(x), x0, x1)
            intQ1 = gquad2(lambda x: (b1 + m1*x)*Q(x), x0, x1)

            # stamp integral componenta for interior nodes
            if i != 0:
                rcd   = add_node(rcd, i, i,     m0*m0*intD)
                rcd   = add_node(rcd, i, i+1,   m0*m1*intD)
                b[i] += intQ0
            if i != self.n-2:
                rcd     = add_node(rcd, i+1, i,   m1*m0*intD)
                rcd     = add_node(rcd, i+1, i+1, m1*m1*intD)
                b[i+1] += intQ1

        # set h on boundaries
        rcd         = add_node(rcd, 0, 0, 1)
        rcd         = add_node(rcd, self.n-1, self.n-1, 1)
        b[0]        = self.hs # south pole
        b[self.n-1] = self.hn # north pole

        # assemble sparse matrix
        A = csc_matrix((rcd[:, 2], (rcd[:, 0], rcd[:, 1])), shape=(self.n, self.n))

        # return as LU factorization
        return splu(A), b

    def solve(self):
        """
            h = self.solve()

        Generate and solve A h = b.
        """
        # generate A and b
        A, b = self._generate_linear_system()

        # solve system
        h = A.solve(b) # J kg-1

        return h
