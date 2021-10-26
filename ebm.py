import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt

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

class EBM():
    """
    Energy Balance Model 

    This model solves the ODE
        -d/dx [ D(x) (1 - x**2) d/dx h(x) ] = Q
    where D is the eddy diffusivity, x = sine latitude, h is moist static energy,
    and Q is the net energy input. The model uses the finite element method. It 
    is designed to take x, Q, and D as inputs and output h.
    """

    def __init__(self, x, Q, D):
        # givens
        self.x = x # grid (sine latitude)
        self.Q = Q # net energy input (W m-2)
        self.D = D # eddy diffusivity (kg m-2 s-1)

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

        # assemble A and b using stamping
        for i in range(self.n-1):
            # interval I = [x0, x1]
            x0 = self.x[i]
            x1 = self.x[i+1]

            # integral of (1 - x**2) on I
            integral = x1 - x1**3/3 - (x0 - x0**3/3)

            # basis vectors on I
            m0, b0 = basis_vector([x0, x1], 0)
            m1, b1 = basis_vector([x0, x1], 1)

            # average D and Q on I
            D = (self.D[i] + self.D[i+1])/2
            Q = (self.Q[i+1] + self.Q[i])/2

            # stamp integral componenta for interior nodes
            if i != 0:
                rcd = add_node(rcd, i, i,     m0*m0*D*integral)
                rcd = add_node(rcd, i, i+1,   m0*m1*D*integral)
                b[i]   += Q*(b0*x1 + m0*x1**2/2 - (b0*x0 + m0*x0**2/2))
            if i != self.n-2:
                rcd = add_node(rcd, i+1, i,   m1*m0*D*integral)
                rcd = add_node(rcd, i+1, i+1, m1*m1*D*integral)
                b[i+1] += Q*(b1*x1 + m1*x1**2/2 - (b1*x0 + m1*x0**2/2))

        # set h on boundaries
        rcd = add_node(rcd, 0, 0, 1)
        rcd = add_node(rcd, self.n-1, self.n-1, 1)
        b[0] = 2.4e6
        b[self.n-1] = 2.4e6

        # assemble sparse matrix
        A = csc_matrix((rcd[:, 2], (rcd[:, 0], rcd[:, 1])), shape=(self.n, self.n))

        # return as LU factorization
        return splu(A), b

    def solve(self):
        """
            h = self.solve()

        Generage and solve A h = b.
        """
        # generate A and b
        A, b = self._generate_linear_system()

        # solve system
        h = A.solve(b) # J kg-1

        return h

# setup test problem
n = 2**8
x = np.linspace(-1, 1, n)
Q = 1365/np.pi*np.cos(np.arcsin(x))
Q -= Q.mean()
D = 2.6e-4*np.ones(n)

# solve
ebm = EBM(x, Q, D)
h = ebm.solve()

# plot
plt.plot(x, h)
plt.show()
