import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt

plt.style.use("plots.mplstyle")

# Constants 
ps = 98000     # kg/m/s2
cp = 1005      # J/kg/K
RH = 0.8       # 0-1
Lv = 2257000   # J/kg

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

# def gquad2(f, a, b):
#     x = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
#     x = (b - a)/2 * x + (a + b)/2
#     return (b - a)/2 * (f(x[0]) + f(x[1]))

# def lerp(p0, p1, x):
#     return p0[1]*(x - p1[0])/(p0[0] - p1[0]) + p1[1]*(x - p0[0])/(p1[0] - p0[0])

def humidsat(t, p):
    """
    FROM BOOS:
    % esat, qsat, rsat = humidsat(t, p)
    %  computes saturation vapor pressure (esat), saturation specific humidity (qsat),
    %  and saturation mixing ratio (rsat) given inputs temperature (t) in K and
    %  pressure (p) in hPa.
    %
    %  these are all computed using the modified Tetens-like formulae given by
    %  Buck (1981, J. Appl. Meteorol.)
    %  for vapor pressure over liquid water at temperatures over 0 C, and for
    %  vapor pressure over ice at temperatures below -23 C, and a quadratic
    %  polynomial interpolation for intermediate temperatures.
    """
    tc=t-273.16;
    tice=-23;
    t0=0;
    Rd=287.04;
    Rv=461.5;
    epsilon=Rd/Rv;

    # first compute saturation vapor pressure over water
    ewat=(1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*tc/(240.97+tc))
    eice=(1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*tc/(272.55+tc))
    # alternatively don"t use enhancement factor for non-ideal gas correction
    #ewat=6.1121.*exp(17.502.*tc./(240.97+tc));
    #eice=6.1115.*exp(22.452.*tc./(272.55+tc));
    eint=eice+(ewat-eice)*((tc-tice)/(t0-tice))**2

    esat=eint
    esat[np.where(tc<tice)]=eice[np.where(tc<tice)]
    esat[np.where(tc>t0)]=ewat[np.where(tc>t0)]

    # now convert vapor pressure to specific humidity and mixing ratio
    rsat=epsilon*esat/(p-esat);
    qsat=epsilon*esat/(p-esat*(1-epsilon));
    return esat, qsat, rsat

def h(T):
    return cp*T + RH*Lv*humidsat(T, ps/100)[1]

def T(h):
    return T_dataset[np.searchsorted(h_dataset, h)]

class EBM():
    """
    Energy Balance Model 

    This model solves the ODE
        -d/dx [ D(x) (1 - x**2) d/dx h(x) ] = Q
    where D is the eddy diffusivity, x = sine latitude, h is moist static energy,
    and Q is the net energy input. The model uses the finite element method. It 
    is designed to take x, Q, and D as inputs and output h.
    """

    def __init__(self, x, Q, D, hs, hn):
        # givens
        self.x = x   # grid (sine latitude)
        self.Q = Q   # net energy input (W m-2)
        self.D = D   # eddy diffusivity (kg m-2 s-1)
        self.hs = hs # south pole moist static energy (J kg-1)
        self.hn = hn # north pole moist static energy (J kg-1)

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
        b[0] = self.hs        # south pole
        b[self.n-1] = self.hn # north pole

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

# T_dataset = np.arange(100, 400, 1e-3)
# q_dataset = humidsat(T_dataset, ps/100)[1]
# h_dataset = cp*T_dataset + RH*q_dataset*Lv

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
