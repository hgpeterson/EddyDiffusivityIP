from sys import modules
import numpy as np
import matplotlib.pyplot as plt
from ebm_LW import EBM
from scipy.linalg import fractional_matrix_power
from scipy.special import legendre
from scipy.optimize import minimize
import os

np.random.seed(42)

plt.style.use("plots.mplstyle")

def data():
    """ Loads observed (or climate model) Moist Static Energy h. 
    Should be 1d arr
    
    Arguments: 
        model (str): CMIP5 model that corresponds to the name of the .npz file
    """
    # TO DO: Add argument for different models
    d = np.load("data/h_Q_CNRM-CM5.npz")
    # d = np.load("data/h_Q_GFDL-CM3.npz")
    # d = np.load("data/h_Q_synthetic.npz")
    h = d["h"]
    x = d["x"]
    S = d["Q_no_TOA_LW"]

    # # add noise to h?
    # h += np.random.normal(loc=0.0, scale=np.mean(h)/100, size=h.shape)

    return h, x, S

def model(x, S, u, spectral):
    """ Define the forward model, or just import it as a separate python 
            function with documented inputs/outputs
    Arguments:
        D (arr): Diffusivity
    """
    # Start with simple model
    ebm = EBM(x, S, u[:-2], u[-2], u[-1], spectral)
    h_tilde = ebm.solve()
    
    return h_tilde, ebm.D

def matrix_norm(A_inv_12, u):
    """ Computes the matrix norm <u, u>_A in the L2 norm.

    Args:
        A_inv_12 (n x n array) 
        u (n x 1 array)

    Returns:
        (float)
    """
    return np.linalg.norm(np.dot(A_inv_12, u))

def log_likelihood(u, gamma_inv_12, h, x, S, spectral):
    # compute negative log likelihood
    h_tilde, D_tilde = model(x, S, u, spectral)
    ll = 0.5*matrix_norm(gamma_inv_12, h - h_tilde)**2

    # non-negative D
    if np.any(D_tilde < 0):
        ll = np.inf

    return ll

def log_likelihoods(gamma_inv_12, u, u_star, h, x, S, spectral):
    """ Computes the ratio of posterior probabilities from u_star/u  
            using observations and forward model evaluation. This provides part
            of the acceptance probability
        Output: 
            move_llikelihood (float): negative log likelihood of the proposed move to u_star
            previous_llikelihood  (float): negative log likelihood of the existing u  
    """
    
    prev_ll = log_likelihood(u, gamma_inv_12, h, x, S, spectral)
    move_ll = log_likelihood(u_star, gamma_inv_12, h, x, S, spectral)
    
    return move_ll, prev_ll

def main():
    # Load data
    h, x, S = data()

    # pre-compute Gamma
    gamma_inv_12 = np.identity(len(x))

    # spectral 
    spectral = True
    # n_polys = 6
    # m = np.zeros(n_polys + 2)
    # m[0] = 0
    # m[1] = 4e-4
    # m[2] = 2e-4
    # m[3] = 2e-4
    # m[4] = 4e-4
    # m[5] = 0

    n_polys = 7
    m = np.zeros(n_polys + 2)
    m[2] = 4e-4
    m[4] = 4e-4

    m[-2] = -1e2
    m[-1] = 2e0

    C = 1e-11*np.identity(len(m))

    C[-2, -2] = 1e-5
    C[-1, -1] = 1e-5

    C_inv_12 = fractional_matrix_power(C, -1/2)

    # number of iterations
    N = 200
    
    # start solution arrays
    us = np.zeros((N + 1, len(m)))
    h_tildes = np.zeros((N + 1, len(x)))
    Ds = np.zeros((N + 1, len(x)))

    # maximum likelihood solution initialized at m
    nll = lambda *args: log_likelihood(*args)
    soln = minimize(nll, m, args=(gamma_inv_12, h, x, S, spectral))
    u = soln.x
    c_hat = 2*soln.hess_inv
    sigma = np.minimum(np.diag(c_hat)**0.5, 5e-5*np.ones(u.size))

    print("Maximum likelihood estimates:")
    print(u)

    print("Sigmas:")
    print(sigma)
    # u = [-1.26738654e-02,1.70174890e-02,-6.47831124e-03,2.34328232e-03,3.49539900e-04,-7.98193981e-03,8.25455115e-03,-2.20258701e+02,1.65905996e+00]

    # Metropolis-Hastings
    for i in range(N): 
        h_tildes[i, :], Ds[i, :] = model(x, S, u, spectral)
        us[i, :] = u

        # draw new parameters
        u_star = np.random.multivariate_normal(u, C)
        
        # compute LLs
        move, prev = log_likelihoods(gamma_inv_12, u, u_star, h, x, S, spectral)

        a = np.min([prev/move, 1])

        if a == 1:
            u = u_star
        elif np.random.uniform() <= a:
            u = u_star
        else:
            u = u

        if i % 100 == 0:
            print(f"{i}/{N}: error = {prev:1.1e}")
        
    # final solution
    h_tildes[N, :], Ds[N, :] = model(x, S, u, spectral)
    us[N, :] = u
    if spectral:
        print(u)

    # save data
    np.savez("out.npz", N=N, x=x, h=h, us=us, h_tildes=h_tildes, Ds=Ds, m=m, C_inv_12=C_inv_12, gamma_inv_12=gamma_inv_12)
    print("out.npz")

def plots():
    # load data
    d= np.load("out.npz")
    N = d["N"]
    x = d["x"]
    h = d["h"]
    us = d["us"]
    h_tildes = d["h_tildes"]
    Ds = d["Ds"]
    m = d["m"]
    C_inv_12 = d["C_inv_12"]
    gamma_inv_12 = d["gamma_inv_12"]

    print(f"A = {us[-1, -2]:1.1e}, B = {us[-1, -1]:1.1e}")
    
    # Plot sum of squared errors in h over time
    fig, ax = plt.subplots()
    errors_h = np.zeros(N)
    # errors_u = np.zeros(N)
    for i in range(1, N+1):
        errors_h[i-1] = 0.5*matrix_norm(gamma_inv_12, h - h_tildes[i, :])**2
        # errors_u[i-1] = 0.5*matrix_norm(C_inv_12, m - us[i, :])**2
    ax.semilogy(errors_h, label=r"$\frac{1}{2} || h - \tilde h ||_\Gamma^2$")
    # ax.semilogy(errors_u, label=r"$\frac{1}{2} || u - m ||_C^2$")
    ax.legend()
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"error")
    # plt.savefig("error.pdf")
    # print("error.pdf")
    plt.savefig("error.png")
    print("error.png")
    plt.close()
        
    # Plot difference between EBM model predicted using our D and the true from the model
    fig, ax = plt.subplots()
    for i in range(9):
        ax.plot(x, h_tildes[int(i*N/9), :]/1e3, c="tab:blue", alpha=i/9)
    ax.plot(x, h_tildes[-1, :]/1e3, c="tab:blue", label=r"$\tilde h$")
    ax.plot(x, h/1e3, c="tab:orange", label=r"$h$")
    ax.set_xlabel(r"latitude $\phi$ (degrees)")
    ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
    ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
    ax.set_ylabel("moist static energy $h$ (kJ kg$^{-1}$)")
    plt.legend()
    # plt.savefig("h.pdf")
    # print("h.pdf")
    plt.savefig("h.png")
    print("h.png")
    plt.close()

    # plot final D
    fig, ax = plt.subplots()
    ax.plot(x, 1e4*Ds[0, :], label="init. cond.")
    ax.plot(x, 1e4*Ds[-1, :], label="inverse result")
    # d = np.load("data/h_Q_synthetic.npz")
    # D_true = d["D"]
    # ax.plot(x, 1e4*D_true, label="truth")
    ax.legend()
    ax.set_xlabel(r"latitude $\phi$ (degrees)")
    ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
    ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
    plt.ylabel(r"diffusivity $D$ ($\times 10^{-4}$ kg m$^{-2}$ s$^{-1}$)")
    # plt.savefig('D.pdf')
    # print("D.pdf")
    plt.savefig('D.png')
    print("D.png")
    plt.close()

if __name__ == '__main__':
    main()
    plots()