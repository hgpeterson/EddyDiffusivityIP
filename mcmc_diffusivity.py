#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Oct 31 17:13:36 2021

@author: rpatel
"""
import numpy as np
import random
import matplotlib.pyplot as plt
from ebm import EBM
import seaborn as sns

def ndiagonal(size_of_a_matrix, diagonal, diagonal1, diagonal2, diagonal3, diagonal4):
    """ Creates a symmetric matrix with n diagonals
    
    Arguments: 
        size_of_a_matrix (int): Creates a nxn matrix using the provided size as n
        diagonal (arr): Array of the diagonal (maybe change this to int, but less flexible??)
        diagonal1 (arr): Array for the first off diagonal, immediately adjacent to the main diagonal
        diagonal2 (arr): Array for the second off diagonal, adjacent to diagonal1s
        diagonal3 (arr): Array for the third off diagonal, adjacent to diagonal2s
    """

   
    matrix = [[0 for j in range(size_of_a_matrix)]
              for i in range(size_of_a_matrix)]
      
    for k in range(size_of_a_matrix-1):
        matrix[k][k] = diagonal[k]
        matrix[k][k+1] = diagonal1[k]
        matrix[k+1][k] = diagonal1[k]
        
        try:
            matrix[k][k+2] = diagonal2[k]
            matrix[k+2][k] = diagonal2[k]
        except IndexError:
            pass
        
        try:
            matrix[k][k+3] = diagonal3[k]
            matrix[k+3][k] = diagonal3[k]
        except IndexError:
            pass
        
        try:
            matrix[k][k+4] = diagonal4[k]
            matrix[k+4][k] = diagonal4[k]
        except IndexError:
            continue


      
    matrix[size_of_a_matrix-1][size_of_a_matrix - 1] = diagonal[size_of_a_matrix-1]
     
    return np.asarray(matrix) 

def data():
    """ Loads observed (or climate model) Moist Static Energy h. 
    Should be 1d arr
    
    Arguments: 
        model (str): CMIP5 model that corresponds to the name of the .npz file
    """
    # TO DO: Add argument for different models
    data = np.load("data/h_Q_CNRM-CM5.npz")
    h = data["h"]
    x = data["x"]
    Q = data["Q"]
    hs = h[0]
    hn = h[-1]

    return h, x, Q, hs, hn

def model(x, Q, D, hs, hn):
    """ Define the forward model, or just import it as a separate python 
            function with documented inputs/outputs
    Arguments:
        D (arr): Diffusivity
    """
    # Start with simple model
    ebm = EBM(x, Q, D, hs, hn)
    h_tilde = ebm.solve()
    
    return np.squeeze(h_tilde)

def log_likelihoods(u, u_star, h, x, Q, hs, hn):
    """ Computes the ratio of posterior probabilities from u_star/u  
            using observations and forward model evaluation. This provides part
            of the acceptance probability
        Output: 
            move_llikelihood (float): negative log likelihood of the proposed move to u_star
            previous_llikelihood  (float): negative log likelihood of the existing u  
    """
    
    # TO DO: Figure out how to define gamma so that we can standardize/weight
    # the observations when taking the weighted Euclidean norm to compute mismatch
    # between observations and model predictions 
#    gamma = np.diag(np.ones(len(x),)*2e+09)  #Example for now, assumes noises are independent, may need tridiagonal here....
    gamma = ndiagonal(len(x), 
                      np.ones(len(x),)*1e+3,
                      np.ones(len(x),)*1e+3*0.9,
                      np.ones(len(x),)*1e+3*0.7,
                      np.ones(len(x),)*1e+3*0.5,
                      np.ones(len(x),)*1e+3*0.3)

        
    # Note: The following formulation does not include a penalty for u being 
    # far away from the prior... unless we get a well-enough defined prior where
    # this is reasonable 
    move_llikelihood = -0.5 * np.sum(np.linalg.inv(gamma**(1/2)) * (h - model(x, Q, u_star, hs, hn))**2)
    previous_llikelihood =  -0.5 * np.sum(np.linalg.inv(gamma**(1/2)) * (h - model(x, Q, u, hs, hn))**2)
    
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


    # Sample from the prior distribution
#    mean = 2.6e-4*np.ones(len(x))   #Started with constant prior
    mean = np.r_[np.linspace(1e-4,4e-4, int(len(x)/4)), 
                 np.linspace(4e-4,1e-4, int(len(x)/4)),
                 np.linspace(1e-4,4e-4, int(len(x)/4)),
                 np.linspace(4e-4,1e-4, int(len(x)/4))]        # Try a different prior, bimodal, or nearly so
    u_arr = mean

#    cov = np.diag(np.ones(180,)*0.1)  # diagonal covariance 
    #(the choice of cov seems EXTREMELY important, or else MCMC won't converge...)
    # If we expect nondiagonal covariance, need to specify it
   
    # Try a sample n-diagonal covariance matrix 
    cov = ndiagonal(len(x),
                    np.ones(len(x),)*1e-11,
                    np.ones(len(x),)*1e-11*0.9,
                    np.ones(len(x),)*1e-11*0.7,
                    np.ones(len(x),)*1e-11*0.5,
                    np.ones(len(x),)*1e-11*0.3)
    u = np.random.multivariate_normal(mean, cov)
    
    for i in range(10000):
        # TO DO: Figure out how to sample from a Markov kernel to get u_star
        # Would this just consider a gaussian with mean of the previous u and then some covariance? 
        u_star = np.random.multivariate_normal(u, cov)
        
        move, previous = log_likelihoods(u, u_star, h, x, Q, hs, hn)
        a = np.min([previous/move, 1])
        
        if a == 1:
            u = u_star
        elif random.uniform(0,1) <= a:
            u = u_star
        else:
            u = u

        u_arr = np.c_[u_arr, u] #Since u is a 1D vector, if we keep this appending method, make sure to append using the correct axis
    

    plt.plot(u_arr[1,:]) #Attempt to plot mixing/convergence at the south pole
    plt.plot(x, u_arr[:,-1]) #Attempt to plot final meridional diffusivity
    
    sns.heatmap(u_arr)
    plt.savefig("D_3p5.png")
    
    for i in range(1,1000,2):
        plt.scatter(i,np.sum((h - model(x, Q, u_arr[:,i], hs, hn))**2))
        
    plt.plot(h-model(x, Q, u_arr[:,1000], hs, hn))
    plt.plot(model(x, Q, u_arr[:,1000], hs, hn))

    # Try making a plot to show uncertainty in D at different latitudes
    lat_med = []
    lat_std = []
    lat_percentile = []
    for i in range(len(x)):
        lat_med = np.append(lat_med, np.median(u_arr[i,-10000:]))
        lat_std = np.append(lat_std, np.std(u_arr[i,-10000:]))
        try:
            lat_percentile.append(np.percentile(u_arr[i,-10000:], [5,25,50,75,95]))
        except:
            lat_percentile.append(np.percentile((np.nan), [5,25,50,75,95]))

    lat_percentile = np.array(lat_percentile)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6, forward=True)
    fig.set_dpi(100)
    ax.fill_between(x, lat_percentile[:,0], lat_percentile[:,4], color='red', alpha=0.15, lw=3)
    ax.fill_between(x, lat_percentile[:,1], lat_percentile[:,3], color='red', alpha=0.35, lw=3)
    ax.plot(x, lat_percentile[:,2], color='k', lw=3)  # Plot the median  data
    #plt.grid(axis='x', color='#f2f2f2')
    plt.grid(axis='y', color='#dedede')
    plt.suptitle('')
    plt.title('Uncertainty in D')
    plt.xlabel('$sin(\phi)$', fontsize=12)
    plt.ylabel('Diffusivity D [kg $m^{-2} s^{-1}$]', fontsize=12, labelpad=0)
    plt.savefig('D_percentiles_3.png', format='png')
    plt.show()

if __name__ == '__main__':
    main()