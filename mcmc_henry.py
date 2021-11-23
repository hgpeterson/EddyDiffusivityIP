import numpy as np
import random
import matplotlib.pyplot as plt
from ebm import EBM
from scipy.linalg import fractional_matrix_power
from scipy.special import legendre
import os

random.seed(42)

plt.style.use("plots.mplstyle")

def data():
    """ Loads observed (or climate model) Moist Static Energy h. 
    Should be 1d arr
    
    Arguments: 
        model (str): CMIP5 model that corresponds to the name of the .npz file
    """
    # TO DO: Add argument for different models
    # data = np.load("data/h_Q_CNRM-CM5.npz")
    d = np.load("data/h_Q_synthetic.npz")
    h = d["h"]
    x = d["x"]
    Q = d["Q"]
    hs = h[0]
    hn = h[-1]

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
    
    return np.squeeze(h_tilde), ebm.D

def matrix_norm(matrix, vector):
    """ Computes the matrix norm <vector, vector>_matrix in the L2 norm.

    Args:
        matrix (n x n array) 
        vector (n x 1 array)

    Returns:
        (float)
    """
    return np.linalg.norm(np.dot(matrix, vector))

def log_likelihoods(gamma_inv_12, C_inv_12, m, u, u_star, h, x, Q, hs, hn, spectral):
    """ Computes the ratio of posterior probabilities from u_star/u  
            using observations and forward model evaluation. This provides part
            of the acceptance probability
        Output: 
            move_llikelihood (float): negative log likelihood of the proposed move to u_star
            previous_llikelihood  (float): negative log likelihood of the existing u  
    """
    
    move = model(x, Q, u_star, hs, hn, spectral)
    prev = model(x, Q, u, hs, hn, spectral)
    move_llikelihood     = -0.5*matrix_norm(gamma_inv_12, h - move[0])**2 \
                           -0.5*matrix_norm(C_inv_12, m - u_star)**2
    previous_llikelihood = -0.5*matrix_norm(gamma_inv_12, h - prev[0])**2 \
                           -0.5*matrix_norm(C_inv_12, m - u)**2

    # non-negative D
    if np.any(move[1] < 0):
        move_llikelihood = -np.inf
    
    return move_llikelihood, previous_llikelihood

def main():
    # Load data
    h, x, Q, hs, hn = data()

    # pre-compute Gamma
    gamma = 1e0*np.identity(len(x))
    gamma_inv_12 = fractional_matrix_power(gamma, -1/2)

    # is gamma positive definite?
    # print(np.all(np.linalg.eigvals(gamma) > 0))

    # spectral prior
    spectral = True
    n_polys = 5
    m = np.zeros(n_polys)
    m[0] = 2.6e-4
    m[4] = -1e-4
    C = 1e-11*np.identity(len(m))
    C_inv_12 = fractional_matrix_power(C, -1/2)

    # # pointwise prior
    # spectral = False
    # # m = 2.6e-4*legendre(0)(x) - 1e-4*legendre(4)(x)
    # m = 2.7e-4*legendre(0)(x) - 0.9e-4*legendre(4)(x)
    # C = 1e-13*np.identity(len(m))
    # C_inv_12 = fractional_matrix_power(C, -1/2)

    # iteration number 
    N = 200
    
    # start solution arrays
    us = np.zeros((N + 1, len(m)))
    h_tildes = np.zeros((N + 1, len(x)))
    Ds = np.zeros((N + 1, len(x)))

    # initialize at mean
    u = m
    
    # main loop
    for i in range(N):
        h_tildes[i, :], Ds[i, :] = model(x, Q, u, hs, hn, spectral)
        us[i, :] = u

        if i % 100 == 0:
            print(f"{i}/{N}")

        u_star = np.random.multivariate_normal(u, C)
        
        move, prev = log_likelihoods(gamma_inv_12, C_inv_12, m, u, u_star, h, x, Q, hs, hn, spectral)
        print(f"{move:1.1e}, {prev:1.1e}")
        a = np.min([prev/move, 1])
        
        if a == 1:
            u = u_star
        elif random.uniform(0,1) <= a:
            u = u_star
        else:
            u = u

    # final solution
    h_tildes[N, :], Ds[N, :] = model(x, Q, u, hs, hn, spectral)
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
    
    # Plot sum of squared errors in h over time
    fig, ax = plt.subplots()
    errors_h = np.zeros(N)
    errors_u = np.zeros(N)
    for i in range(1, N+1):
        errors_h[i-1] = -0.5*matrix_norm(gamma_inv_12, h - h_tildes[i, :])**2
        errors_u[i-1] = -0.5*matrix_norm(C_inv_12, m - us[i, :])**2
    ax.semilogy(np.abs(errors_h), label=r"$\frac{1}{2} || h - \tilde h ||_\Gamma^2$")
    ax.semilogy(np.abs(errors_u), label=r"$\frac{1}{2} || u - m ||_C^2$")
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
    ax.plot(x, h/1e3, label="reference model")
    ax.plot(x, h_tildes[-1, :]/1e3, label="EBM")
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
    d = np.load("data/h_Q_synthetic.npz")
    D_true = d["D"]
    ax.plot(x, 1e4*D_true, label="reference model")
    ax.plot(x, 1e4*Ds[-1, :], label="inverse result")
    ax.plot(x, 1e4*Ds[0, :], "--", label="init. cond.")
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