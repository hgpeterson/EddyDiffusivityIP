from sys import modules
import numpy as np
import matplotlib.pyplot as plt
from ebm import EBM
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
    # d = np.load("data/h_Q_CNRM-CM5.npz")
    # d = np.load("data/h_Q_synthetic.npz")
    d = np.load("data/h_Q_CAN-ESM2.npz")
    h = d["h"]
    x = d["x"]
    Q = d["Q"]
    hs = h[0]
    hn = h[-1]

    # # add noise to h?
    # h += np.random.normal(loc=0.0, scale=np.mean(h)/100, size=h.shape)

    return h, x, Q, hs, hn

def model(x, Q, D, hs, hn, spectral):
    """ Define the forward model, or just import it as a separate python 
            function with documented inputs/outputs
    Arguments:
        D (arr): Diffusivity
    """
    # Start with simple model
    ebm = EBM(x, Q, D, hs, hn, spectral)
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

def log_likelihood(u, gamma_inv_12, h, x, Q, hs, hn, spectral):
    # compute negative log likelihood
    h_tilde, D_tilde = model(x, Q, u, hs, hn, spectral)
    ll = 0.5*matrix_norm(gamma_inv_12, h - h_tilde)**2

    # non-negative D
    if np.any(D_tilde < 0):
        ll = np.inf

    return ll

def log_likelihoods(gamma_inv_12, u, u_star, h, x, Q, hs, hn, spectral):
    """ Computes the ratio of posterior probabilities from u_star/u  
            using observations and forward model evaluation. This provides part
            of the acceptance probability
        Output: 
            move_llikelihood (float): negative log likelihood of the proposed move to u_star
            previous_llikelihood  (float): negative log likelihood of the existing u  
    """
    
    prev_ll = log_likelihood(u, gamma_inv_12, h, x, Q, hs, hn, spectral)
    move_ll = log_likelihood(u_star, gamma_inv_12, h, x, Q, hs, hn, spectral)
    
    return move_ll, prev_ll

def main():
    # Load data
    h, x, Q, hs, hn = data()

    # pre-compute Gamma
    gamma_inv_12 = np.identity(len(x))

    # spectral 
    spectral = True
    n_polys = 10
    # n_polys = 5
    u0 = np.zeros(n_polys)
    # u0[0] = 4e-4
    u0[0] = 2.6e-4
    u0[4] = -1e-4
    C = 1e-11*np.identity(len(u0))

    # iterations 
    N_ml = 2000
    N_mcmc = 4000
    
    # initialize
    u = u0
    h0, D0 = model(x, Q, u, hs, hn, spectral)
    
    # first loop: Maximum Likelihood
    for i in range(N_ml):
        # draw new parameters
        u_star = np.random.multivariate_normal(u, C)
        
        # compute LLs
        move, prev = log_likelihoods(gamma_inv_12, u, u_star, h, x, Q, hs, hn, spectral)

        # only pick solutions that reduce the LL
        if move < prev:
            u = u_star

        if i % 100 == 0:
            print(f"{i}/{N_ml}: error = {prev:1.1e}")

    if spectral:
        print("Maximum-Likelihood: ", u)

    # start solution arrays
    us = np.zeros((N_mcmc + 1, len(u)))
    h_tildes = np.zeros((N_mcmc + 1, len(x)))
    Ds = np.zeros((N_mcmc + 1, len(x)))

    # different C for MCMC
    C = 1e-13*np.identity(len(u))

    # second loop: Metropolis-Hastings
    for i in range(N_mcmc): 
        h_tildes[i, :], Ds[i, :] = model(x, Q, u, hs, hn, spectral)
        us[i, :] = u

        # draw new parameters
        u_star = np.random.multivariate_normal(u, C)
        
        # compute LLs
        move, prev = log_likelihoods(gamma_inv_12, u, u_star, h, x, Q, hs, hn, spectral)

        a = np.min([prev/move, 1])

        if a == 1:
            u = u_star
        elif np.random.uniform() <= a:
            u = u_star
        else:
            u = u

        if i % 100 == 0:
            print(f"{i}/{N_mcmc}: error = {prev:1.1e}")
        
    # final solution
    h_tildes[N_mcmc, :], Ds[N_mcmc, :] = model(x, Q, u, hs, hn, spectral)
    us[N_mcmc, :] = u
    if spectral:
        print("MCMC:", u)

    # save data
    np.savez("out.npz", N=N_mcmc, x=x, h=h, h0=h0, us=us, h_tildes=h_tildes, Ds=Ds, D0=D0, u0=u0, gamma_inv_12=gamma_inv_12)
    print("out.npz")

def plots():
    # load data
    d= np.load("out.npz")
    N = d["N"]
    x = d["x"]
    h = d["h"]
    h0 = d["h0"]
    us = d["us"]
    h_tildes = d["h_tildes"]
    Ds = d["Ds"]
    u0 = d["u0"]
    D0 = d["D0"]
    gamma_inv_12 = d["gamma_inv_12"]
    
    # Plot sum of squared errors in h over time
    fig, ax = plt.subplots()
    errors_h = np.zeros(N)
    for i in range(1, N+1):
        errors_h[i-1] = 0.5*matrix_norm(gamma_inv_12, h - h_tildes[i, :])**2
    ax.semilogy(errors_h, label=r"$\frac{1}{2} || h - \tilde h ||_\Gamma^2$")
    ax.legend()
    ax.set_xlabel("iteration")
    ax.set_ylabel("error")
    plt.savefig("error.png")
    print("error.png")
    plt.close()
        
    # Plot difference between EBM model predicted using our D and the true from the model
    fig, ax = plt.subplots()
    ax.plot(x, h0/1e3, c="gray", label="initial guess")
    h_max = np.max(h_tildes, axis=0)/1e3
    h_min = np.min(h_tildes, axis=0)/1e3
    ax.fill_between(x, h_min, h_max, color="tab:blue", alpha=0.5, lw=0)
    ax.plot(x, h_tildes[-1, :]/1e3, c="tab:blue", label="mcmc")
    ax.plot(x, h_tildes[0, :]/1e3, "k--", label="max likelihood")
    ax.plot(x, h/1e3, c="tab:orange", label="truth")
    ax.set_xlabel(r"latitude $\phi$ (degrees)")
    ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
    ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
    ax.set_ylabel("moist static energy $h$ (kJ kg$^{-1}$)")
    plt.legend()
    plt.savefig("h.png")
    print("h.png")
    plt.close()

    # plot final D
    fig, ax = plt.subplots()
    ax.plot(x, 1e4*D0, c="gray", label="initial guess")
    D_max = 1e4*np.max(Ds, axis=0)
    D_min = 1e4*np.min(Ds, axis=0)
    ax.fill_between(x, D_min, D_max, color="tab:blue", alpha=0.5, lw=0)
    ax.plot(x, 1e4*Ds[-1, :], c="tab:blue", label="mcmc")
    ax.plot(x, 1e4*Ds[0, :], "k--", label="max likelihood")
    # d = np.load("data/h_Q_synthetic.npz")
    # D_true = d["D"]
    # ax.plot(x, 1e4*D_true, c="tab:orange", label="truth")
    ax.legend()
    ax.set_xlabel(r"latitude $\phi$ (degrees)")
    ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
    ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
    plt.ylabel(r"diffusivity $D$ ($\times 10^{-4}$ kg m$^{-2}$ s$^{-1}$)")
    plt.savefig('D.png')
    print("D.png")
    plt.close()

def plots_report():
    # load data
    d= np.load("sims/synth_noise/out.npz")
    N = d["N"]
    x = d["x"]
    h = d["h"]
    h0 = d["h0"]
    us = d["us"]
    h_tildes = d["h_tildes"]
    Ds = d["Ds"]
    u0 = d["u0"]
    D0 = d["D0"]
    gamma_inv_12 = d["gamma_inv_12"]
    
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 6.5/1.62/2))
    ax[0].plot(x, h0/1e3, c="gray", label="initial guess")
    h_max = np.max(h_tildes, axis=0)/1e3
    h_min = np.min(h_tildes, axis=0)/1e3
    ax[0].fill_between(x, h_min, h_max, color="tab:blue", alpha=0.5, lw=0)
    ax[0].plot(x, h_tildes[-1, :]/1e3, c="tab:blue", label="mcmc")
    ax[0].plot(x, h_tildes[0, :]/1e3, "k--", label="max likelihood")
    ax[0].plot(x, h/1e3, c="tab:orange", label="truth")
    ax[0].set_xlabel(r"latitude $\phi$ (degrees)")
    ax[0].set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
    ax[0].set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
    ax[0].set_ylabel("moist static energy $h$ (kJ kg$^{-1}$)")
    ax[0].legend()

    ax[1].plot(x, 1e4*D0, c="gray", label="initial guess")
    D_max = 1e4*np.max(Ds, axis=0)
    D_min = 1e4*np.min(Ds, axis=0)
    ax[1].fill_between(x, D_min, D_max, color="tab:blue", alpha=0.5, lw=0)
    ax[1].plot(x, 1e4*Ds[-1, :], c="tab:blue", label="mcmc")
    ax[1].plot(x, 1e4*Ds[0, :], "k--", label="max likelihood")
    d = np.load("data/h_Q_synthetic.npz")
    D_true = d["D"]
    ax[1].plot(x, 1e4*D_true, c="tab:orange", label="truth")
    ax[1].set_xlabel(r"latitude $\phi$ (degrees)")
    ax[1].set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
    ax[1].set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
    plt.ylabel(r"diffusivity $D$ ($\times 10^{-4}$ kg m$^{-2}$ s$^{-1}$)")

    ax[0].annotate("(a)", (-0.05, 1.05), xycoords="axes fraction")
    ax[1].annotate("(b)", (-0.05, 1.05), xycoords="axes fraction")

    plt.savefig('synth_mcmc.pdf')
    print("synth_mcmc.pdf")
    plt.close()

if __name__ == '__main__':
    # main()
    # plots()
    plots_report()