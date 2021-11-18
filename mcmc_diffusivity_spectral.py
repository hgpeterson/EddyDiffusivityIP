import numpy as np
import random
import matplotlib.pyplot as plt
from ebm import EBM
from scipy.linalg import fractional_matrix_power

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

def model(x, Q, D, hs, hn, spectral=True):
    """ Define the forward model, or just import it as a separate python 
            function with documented inputs/outputs
    Arguments:
        D (arr): Diffusivity
    """
    # Start with simple model
    ebm = EBM(x, Q, D, hs, hn, spectral)
    h_tilde = ebm.solve()
    D = ebm.D
    
    return np.squeeze(h_tilde), D

def log_likelihoods(gamma_inv_12, u, u_star, h, x, Q, hs, hn):
    """ Computes the ratio of posterior probabilities from u_star/u  
            using observations and forward model evaluation. This provides part
            of the acceptance probability
        Output: 
            move_llikelihood (float): negative log likelihood of the proposed move to u_star
            previous_llikelihood  (float): negative log likelihood of the existing u  
    """
        
    # Note: The following formulation does not include a penalty for u being 
    # far away from the prior... unless we get a well-enough defined prior where
    # this is reasonable 
    move_llikelihood     = -0.5 * np.sum(np.dot(gamma_inv_12, (h - model(x, Q, u_star, hs, hn)[0])))**2
    previous_llikelihood = -0.5 * np.sum(np.dot(gamma_inv_12, (h - model(x, Q, u, hs, hn)[0])))**2
    
    return move_llikelihood, previous_llikelihood


def main():
    # TO DO:
    # 1) Initalize markov kernel from a multivariate Gaussian with constant mean and cleverly chosen covariance   
    # 2) Propose a new parameter u*_(n+1)
    # 3) Set u_(n+1) = u∗_(n+1) with probability a(un, u∗_(n+1)); otherwise set u_(n+1) = u_n
    #   Consider definining the acceptance probability like Eq 2.15 in https://clima.caltech.edu/files/2020/01/Calibrate-Emulate-Sample-2021.pdf 
    #   This is also just the log likelihood, evaluating with a forward model
    # 4) After the burn-in period, consider the distribution of 1000+ samples to estimate u and do UQ
    # 5) Visualize the distribution of u (diffusivity) lat on abscissa, D on ordinate (or flipped, idc), and heatmap-like shading based on density of samples
    
    # Load data
    h, x, Q, hs, hn = data()

    # pre-compute Gamma
    gamma = np.identity(len(x))
    gamma_inv_12 = fractional_matrix_power(gamma, -1/2)

    # is gamma positive definite?
    # print(np.all(np.linalg.eigvals(gamma) > 0))

    # Sample from the prior distribution
    n_polys = 5
    mean = np.zeros(n_polys)
    mean[0] = 2.6e-4
    mean[4] = -1e-4
    u_arr = mean

    # Try a sample n-diagonal covariance matrix 
    cov = 1e-10*np.identity(len(mean))
    u = np.random.multivariate_normal(mean, cov)
    
    N = 1000
    h_tildes = np.zeros((N, len(x)))
    Ds = np.zeros((N, len(x)))
    for i in range(N):
        # TO DO: Figure out how to sample from a Markov kernel to get u_star
        # Would this just consider a gaussian with mean of the previous u and then some covariance? 
        
        h_tildes[i, :], Ds[i, :] = model(x, Q, u, hs, hn)

        if i % 100 == 0:
            print(f"{i}/{N}")

        u_star = np.random.multivariate_normal(u, cov)
        
        move, previous = log_likelihoods(gamma_inv_12, u, u_star, h, x, Q, hs, hn)
        a = np.min([previous/move, 1])

        print(f"{previous:1.1e}")
        
        if a == 1:
            u = u_star
        elif random.uniform(0,1) <= a:
            u = u_star
        else:
            u = u

        u_arr = np.c_[u_arr, u] #Since u is a 1D vector, if we keep this appending method, make sure to append using the correct axis

    # print final solution
    print(u)
    
    # Plot sum of squared errors in h over time
    fig, ax = plt.subplots()
    errors = np.zeros(N)
    for i in range(1, N):
        errors[i] = np.sum(np.dot(gamma_inv_12, (h - h_tildes[i, :])))**2
    ax.semilogy(np.abs(errors))
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"model error $|| y - G(u) ||_\Gamma^2$")
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
