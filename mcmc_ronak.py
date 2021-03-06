from sys import modules
import numpy as np
import matplotlib.pyplot as plt
from ebm import EBM
from scipy.linalg import fractional_matrix_power
from scipy.special import legendre
import os

np.random.seed(64)

plt.style.use("plots.mplstyle")

def data(mdl):
    """ Loads observed (or climate model) Moist Static Energy h. 
    Should be 1d arr
    
    Arguments: 
        mdl (str): CMIP5 model that corresponds to the name of the .npz file
    """
    # TO DO: Add argument for different models
    d = np.load("data/" + str(mdl) + ".npz")
    # d = np.load("data/h_Q_synthetic.npz")
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
                          + 0.5*matrix_norm(1e7, m - u_star)**2  #First tried the matrix norm of C_inv_12, but the orders of magnitude between that and h are vastly different.
    previous_llikelihood = 0.5*matrix_norm(gamma_inv_12, h - prev[0])**2 \
                          + 0.5*matrix_norm(1e7, m - u)**2

    # non-negative D
    if np.any(move[1] < 0):
        move_llikelihood = np.inf
    
    return move_llikelihood, previous_llikelihood

def main(mdl):
    # Load data
    h, x, Q, hs, hn = data(mdl)

    # pre-compute Gamma
    gamma = 1e0*np.identity(len(x))
    gamma_inv_12 = fractional_matrix_power(gamma, -1/2)

    # is gamma positive definite?
    # print(np.all(np.linalg.eigvals(gamma) > 0))

    # spectral prior
    spectral = True
    # n_polys = 5
    n_polys = 10
    m = np.zeros(n_polys)
    m[0] = 2.6e-4
    m[4] = -1e-4
    C = 1e-12*np.identity(len(m))
    C_inv_12 = fractional_matrix_power(C, -1/2)

    # # pointwise prior
    # spectral = False
    # # m = 2.6e-4*legendre(0)(x) - 1e-4*legendre(4)(x)
    # m = 2.7e-4*legendre(0)(x) - 0.9e-4*legendre(4)(x)
    # C = 1e-13*np.identity(len(m))
    # C_inv_12 = fractional_matrix_power(C, -1/2)

    # iteration number 
    N = 2500
    
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

        # draw new parameters
        u_star = np.random.multivariate_normal(u, C)
        
        # compute llikelihoods
        move, prev = log_likelihoods(gamma_inv_12, C_inv_12, m, u, u_star, h, x, Q, hs, hn, spectral)
        a = np.min([prev/move, 1])

        if a == 1:
            u = u_star
#        elif np.random.uniform() <= a:
#            u = u_star
        else:
            u = u

        if i % 100 == 0:
            print(f"{i}/{N}: error = {prev:1.1e}")
        
    # Now use this u as a prior and find covariance from this to do actual MCMC
    
    # spectral prior
    spectral = True
    n_polys = 10
    m = u

    #TO DO: If needed, set covariance matrix using inverse of Hessian of last iterations from initial MCMC run
    C = 1e-12*np.identity(len(m))
    C_inv_12 = fractional_matrix_power(C, -1/2)

    # iteration number 
    N = 2500
    
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

        # draw new parameters
        u_star = np.random.multivariate_normal(u, C)
        
        # compute llikelihoods
        move, prev = log_likelihoods(gamma_inv_12, C_inv_12, m, u, u_star, h, x, Q, hs, hn, spectral)
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
    h_tildes[N, :], Ds[N, :] = model(x, Q, u, hs, hn, spectral)
    us[N, :] = u
    if spectral:
        print(u)

    # save data
    np.savez("out_" + str(mdl) + ".npz", N=N, x=x, h=h, us=us, h_tildes=h_tildes, Ds=Ds, m=m, C_inv_12=C_inv_12, gamma_inv_12=gamma_inv_12)
    print("out.npz")

def plots(model):
    # load data
    d= np.load("out_" + str(mdl) + ".npz")
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
        errors_h[i-1] = 0.5*matrix_norm(gamma_inv_12, (h - h_tildes[i, :])/1e6)**2
        errors_u[i-1] = 0.5*matrix_norm(C_inv_12, m - us[i, :])**2
    ax.semilogy(errors_h, label=r"$\frac{1}{2} || h - \tilde h ||_\Gamma^2$")
    # ax.semilogy(errors_u, label=r"$\frac{1}{2} || u - m ||_C^2$")
    ax.legend()
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"error")
    # plt.savefig("error.pdf")
    # print("error.pdf")
    plt.savefig("error_" + str(model) + ".png")
    print("error.png")
    plt.close()
        
    # Plot difference between EBM model predicted using our D and the true from the model
    fig, ax = plt.subplots()
    ax.plot(x, h/1e3, label="reference model")
    for i in range(9):
        ax.plot(x, h_tildes[int(i*N/9), :]/1e3, c="tab:orange", alpha=i/9)
    ax.plot(x, h_tildes[-1, :]/1e3, c="tab:orange", label="EBM")
    ax.set_xlabel(r"latitude $\phi$ (degrees)")
    ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
    ax.set_xticklabels(["90??S", "", "", "", "", "", "30??S", "", "", "EQ", "", "", "30??N", "", "", "", "", "", "90??N"])
    ax.set_ylabel("moist static energy $h$ (kJ kg$^{-1}$)")
    plt.legend()
    # plt.savefig("h.pdf")
    # print("h.pdf")
    plt.savefig("h_" + str(model) + ".png")
    print("h.png")
    plt.close()

    # plot final D
    fig, ax = plt.subplots()
    # d = np.load("data/h_Q_synthetic.npz")
    # D_true = d["D"]
    # ax.plot(x, 1e4*D_true, label="reference model")
    ax.plot(x, 1e4*Ds[-1, :], label="inverse result")
    ax.plot(x, 1e4*Ds[0, :], "--", label="init. cond.")
    ax.legend()
    ax.set_xlabel(r"latitude $\phi$ (degrees)")
    ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
    ax.set_xticklabels(["90??S", "", "", "", "", "", "30??S", "", "", "EQ", "", "", "30??N", "", "", "", "", "", "90??N"])
    plt.ylabel(r"diffusivity $D$ ($\times 10^{-4}$ kg m$^{-2}$ s$^{-1}$)")
    # plt.savefig('D.pdf')
    # print("D.pdf")
    plt.savefig('D_' + str(model) + '.png')
    print("D.png")
    plt.close()
    
    # plot uncertainty in D
    lat_med = []
    lat_std = []
    lat_percentile = []
    for i in range(len(x)):
        lat_med = np.append(lat_med, np.median(Ds[:, i]))
        lat_std = np.append(lat_std, np.std(Ds[:, i]))
        try:
            lat_percentile.append(np.percentile(Ds[:, i], [5,25,50,75,95]))
        except:
            lat_percentile.append(np.percentile((np.nan), [5,25,50,75,95]))

    lat_percentile = np.array(lat_percentile)

    fig, ax = plt.subplots()
    ax.fill_between(x, lat_percentile[:,0], lat_percentile[:,4], color='red', alpha=0.15, lw=3)
    ax.fill_between(x, lat_percentile[:,1], lat_percentile[:,3], color='red', alpha=0.35, lw=3)
    ax.plot(x, lat_percentile[:,2], color='k', lw=3)  # Plot the median  data
    #plt.grid(axis='x', color='#f2f2f2')
    plt.grid(axis='y', color='#dedede')
    plt.suptitle('')
    ax.set_xlabel(r"latitude $\phi$ (degrees)")
    ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
    ax.set_xticklabels(["90??S", "", "", "", "", "", "30??S", "", "", "EQ", "", "", "30??N", "", "", "", "", "", "90??N"])
    plt.ylabel(r"diffusivity $D$ ($\times 10^{-4}$ kg m$^{-2}$ s$^{-1}$)")
    plt.title('Uncertainty in D')
    plt.ylabel('Diffusivity D [kg $m^{-2} s^{-1}$]', fontsize=12, labelpad=0)
    plt.tight_layout()
    plt.savefig('D_percentiles_new_' + str(model) + '.png', format='png')
    plt.show()


if __name__ == '__main__':
    for mdl in ["h_Q_ACCESS1-0", "h_Q_ACCESS1-3", "h_Q_GFDL-ESM2G", "h_Q_GFDL-ESM2M", "h_Q_IPSL-CM5ALR", "h_Q_MPI-ESMLR", "h_Q_MRI-CGCM3", "h_Q_NOR-ESM1M" ]:
#    for mdl in ["h_Q_CAN-ESM2" ]:
#    for mdl in ["h_Q_CAN-ESM2", "h_Q_CNRM-CM5", "h_Q_GFDL-CM3", "h_Q_MIROC-ESM", "h_Q_INM-CM4"]:
        main(mdl)
        plots(mdl)
        