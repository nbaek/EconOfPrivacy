import numpy as np
from numpy.polynomial.hermite import hermgauss
import matplotlib.pyplot as plt

def GaussHermite_lognorm(sigma, n):
    x, w = hermgauss(n)
    x = np.exp(x*np.sqrt(2)*sigma - 0.5*(sigma**2))
    w = w/np.sqrt(np.pi)
    return x, w

def nonlinspace(lo, up, n, phi):
    """
    Function to create a nonlinear space.
    Used for optimization for particular areas in a grid.
    Input:
        lo := Lower bound (list or integer)
        up := upper bound (list or integer)
        n  := no. of gridpoints (integer)
        phi:= non-linearity parameter. of >1 more wieght on lower part
    
    Output:
        x := nonlinear grid of n points. Type is numpy.ndarray
    
    Dimensions: If lo is vector then x is a matrix.
    Requires numpy package to be loaded in before function is run.
    """
    x = np.zeros([n, np.array([lo]).size])
    x[:,:] = np.nan
    x[0,:] = np.array([lo])
    hi = np.array(up, dtype=float)
    for i in range(1,n):
        x[i,:] = x[i-1,:] + (hi-x[i-1,:])/((n-i)**phi)
    
    return x

def figConsumption(par, sol, zeit = [0, 10, 20, 30, 40, 50]):
    """
    Draws consomption function for different time periods
    Input:
        par: model paramters (class)
        sol: Solutions to the model (dict)
        zeit: timeperiods to plot (T - zeit) (list) 
    Output:
        Printet figures
    """
    plt.figure()
    for t in zeit:
        tid = par.T - t
        plt.plot(sol['m'][tid], sol['c'][tid], '-', label='t = ' + str(tid))
        plt.title('Consumption function for periods')
    plt.show()

