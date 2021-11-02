#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Oct 31 17:13:36 2021

@author: rpatel
"""
import numpy as np
import random
import matplotlib.pyplot as plt

def data():
    """ Loads observed (or climate model) Moist Static Energy h. 
    Should be 1d arr
    """
    #TO DO: Obtain ERA5/Climate model data for h, then save to csv/txt to read?
    y = np.ones(180,)*7

    return y

def model(u):
    """ Define the forward model, or just import it as a separate python 
            function with documented inputs/outputs
       Assumes outputs h and D is at least 1 input.  
    """
    # Start with simple model
    y = 7*u
    
    return y

def log_likelihoods(u, u_star):
    """ Computes the ratio of posterior probabilities from u_star/u  
            using observations and forward model evaluation. This provides part
            of the acceptance probability
            Output: Either ratio of log-likelihoods, 
                or if we refactor this function to more generically produce 
                neg log likelihoods and just take the appropriate ratio in
                main or something
    """
    y = data()
    
    # TO DO: Figure out how to define gamma so that we can standardize/weight
    # the observations when taking the weighted Euclidean norm to compute mismatch
    # between observations and model predictions 
    gamma = np.diag(np.ones(180,)*1.5)  #Example for now
        
    # Note: The following formulation does not include a penalty for u being 
    # far away from the prior... unless we get a well-enough defined prior where
    # this is reasonable 
    move_llikelihood = 0.5 * np.sum(np.linalg.inv(gamma**(1/2)) * (y - model(u_star))**2)
    previous_llikelihood =  0.5 * np.sum(np.linalg.inv(gamma**(1/2)) * (y - model(u))**2)
    
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
    
    
    # Sample from the prior distribution
    mean = np.ones(180,)*9.5
    u_arr = mean

#    cov = np.diag(np.ones(180,)*0.1)  # diagonal covariance 
    #(the choice of cov seems EXTREMELY important, or else MCMC won't converge...)
    # If we expect nondiagonal covariance, need to specify it
   
    # Example, if we know all points will have to be the same, bc of the way 
    # this toy example is, made the cov matrix all 0.1s. It converges to the correct value
    cov = np.ones((180,180))*0.3
    u = np.random.multivariate_normal(mean, cov)
    
    for i in range(2000):
        # TO DO: Figure out how to sample from a Markov kernel to get u_star
        # Would this just consider a gaussian with mean of the previous u and then some covariance? 
        u_star = np.random.multivariate_normal(u, cov)
        
        move, previous = log_likelihoods(u, u_star)
        a = np.min([previous/move, 1])
        
        if a == 1:
            u = u_star
        elif random.uniform(0,1) <= a:
            u = u_star
        else:
            u = u

        u_arr = np.c_[u_arr, u] #Since u is a 1D vector, if we keep this appending method, make sure to append using the correct axis
    
    plt.plot(u_arr[1])
    plt.xlim(0,10)
    
if __name__ == '__main__':
    main()