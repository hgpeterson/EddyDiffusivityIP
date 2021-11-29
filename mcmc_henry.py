from sys import modules
import numpy as np
import matplotlib.pyplot as plt
from ebm import EBM
from scipy.linalg import fractional_matrix_power
from scipy.special import legendre
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
    # d = np.load("data/h_Q_synthetic.npz")
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
    move_llikelihood     = 0.5*matrix_norm(gamma_inv_12, h - move[0])**2 \
                        #  + 0.5*matrix_norm(C_inv_12, m - u_star)**2
    previous_llikelihood = 0.5*matrix_norm(gamma_inv_12, h - prev[0])**2 \
                        #  + 0.5*matrix_norm(C_inv_12, m - u)**2

    # non-negative D
    if np.any(move[1] < 0):
        move_llikelihood = np.inf
    
    return move_llikelihood, previous_llikelihood

def main():
    # Load data
    h, x, Q, hs, hn = data()

    # pre-compute Gamma
    gamma_inv_12 = np.identity(len(x))

    # spectral 
    spectral = True
    n_polys = 10
    m = np.zeros(n_polys)
    # m[0] = 2.6e-4
    m[0] = 3.1e-4
    m[4] = -1e-4
    # m = np.array([2.64097918e-04, -2.16454667e-06, -1.03828721e-05, -9.75962728e-06, -8.66203104e-05])
    # m = np.array([ 3.17431465e-04 -4.83683689e-05  8.23995368e-05  9.01738600e-05 8.49767620e-06 -1.46483938e-05  6.31743543e-05 -2.40702704e-05 5.70811678e-05 -7.10327860e-06])
    C = 1e-13*np.identity(len(m))
    C_inv_12 = fractional_matrix_power(C, -1/2)

    # # pointwise 
    # spectral = False
    # m = 3.1e-4*legendre(0)(x) - 1e-4*legendre(4)(x)
    # C = 1e-11*np.identity(len(m))
    # C_inv_12 = fractional_matrix_power(C, -1/2)

    # iteration number 
    N = 4000
    
    # start solution arrays
    us = np.zeros((N + 1, len(m)))
    h_tildes = np.zeros((N + 1, len(x)))
    Ds = np.zeros((N + 1, len(x)))

    # initialize at mean
    u = m
    
    # first loop: only pick solutions that reduce the LL
    for i in range(N//2):
        h_tildes[i, :], Ds[i, :] = model(x, Q, u, hs, hn, spectral)
        us[i, :] = u

        # draw new parameters
        u_star = np.random.multivariate_normal(u, C)
        
        # compute LLs
        move, prev = log_likelihoods(gamma_inv_12, C_inv_12, m, u, u_star, h, x, Q, hs, hn, spectral)

        # only pick solutions that reduce the LL
        if move < prev:
            u = u_star

        if i % 100 == 0:
            print(f"{i}/{N}: error = {prev:1.1e}")

    if spectral:
        print(u)

    # empirical covariance
    # C = np.cov(us[N//4:N//2, :].T)
    C = 1e-13*np.identity(len(u))
    C_inv_12 = fractional_matrix_power(C, -1/2)
    # fig, ax = plt.subplots()
    # vmax = np.max(np.abs(C))
    # plt.imshow(C, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    # plt.colorbar()
    # plt.savefig("C.png")
    # plt.close()

    # second loop: Metropolosi-Hastings
    for i in range(N//2, N): 
        h_tildes[i, :], Ds[i, :] = model(x, Q, u, hs, hn, spectral)
        us[i, :] = u

        # draw new parameters
        u_star = np.random.multivariate_normal(u, C)
        
        # compute LLs
        move, prev = log_likelihoods(gamma_inv_12, C_inv_12, m, u, u_star, h, x, Q, hs, hn, spectral)

        a = np.min([prev/move, 1])

        if a == 1:
            u = u_star
        elif np.random.uniform() <= a and i > 1000: # just prioritize minimizing LL for the first 1000 iterations
            u = u_star
        else:
            u = u

        if i % 100 == 0:
            print(f"{i}/{N}: error = {prev:1.1e}")
        
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