import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel

def explore_lorentzian_pars(x,y):
    """
    Find the maximum value of the y-data. 
    Find the x-coordinate corresponding to the maximum y-data value
    find the width corresponding to the points on the x-axis which are 1/2 the value of the maximum y-value

    Parameters
    ----------
    x: numpy array
        x-data

    y: numpy array
        y-data

    """  
    
    fig, ax = plt.subplots(figsize = (8,6))
    
    ax.plot(x,y)
    
    maxval = y.max()
    halfmax = maxval / 2
    maxpos = y.argmax()
    leftpos = (np.abs(y[:maxpos] - halfmax)).argmin()
    rightpos = (np.abs(y[maxpos:] - halfmax)).argmin() + maxpos
    fwhm = x[rightpos] - x[leftpos]
    ax.hlines(halfmax, x[leftpos], x[rightpos], color='crimson', ls=':')
    
    ax.text(x[maxpos], halfmax, f'fwhm (W)= {fwhm:.3f}\n', ha='center', va='center',fontsize = 20)
    ax.text(x[maxpos], maxval, f'Max y-val (A)= {maxval:.3f}\n', ha='center', va='center',fontsize = 20)
    ax.text(x[maxpos], 0, f'x-val at Max (C)= {x[maxpos]:.3f}\n', ha='center', va='center',fontsize = 20)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(24)
    ax.set_ylim([0,maxval])

def gauss_hist(data_array,n_bins = 50,plotflag = True,return_result = False):
    """
    Break a 1-d array into n_bins and then fit a gaussian to them

    Parameters
    ----------
    data_array: numpy array
        data array to be binned
    
    n_bins: int
        number of bins to break data_array into

    plotflag: bool
        Option to plot the results of the fit
    
    return_result: bool
        Option to return the ModelResult object from gaussian fitting


    """  
    bin_heights, bin_borders = np.histogram(data_array,bins = n_bins)
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2

    x = bin_centers
    y = bin_heights

    mod = GaussianModel()

    pars = mod.guess(y, x=x)
    out = mod.fit(y, pars, x=x)


    if plotflag:
        
        print(out.fit_report(min_correl=0.25))
        w = bin_centers[1] - bin_centers[0]

        fig,ax = plt.subplots(figsize = (12,8))
        ax.bar(bin_centers,bin_heights,width = w*0.9)
        ax = out.plot_fit()
        for item in ([ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)
        
        bin_range = np.abs(np.max(bin_centers) - np.min(bin_centers))
        ax.set_xlim([np.min(bin_centers) - bin_range*0.5,np.max(bin_centers)+bin_range*0.5])
    
    if return_result:
        return out    
    
def check_chi2(FObject,idx,variance):
    """Get the chi-2 value of the fit result

    Parameters
    ----------
    FObject: Function Object
    
    idx: int
        dataset index to get chi-squared value from

    variance: float
        The variance used to generate the noise 
    
    """
    data = FObject.data[idx]
    fit_line = FObject.mod.eval(params = FObject.fit_results[idx].params,x = FObject.x)

    return np.sum(((fit_line - data)/variance)**2)    
    
def check_pars1d(FObject,idx = 0,scale = 0.5):
    """
    Visually check the parameter space aroudn fit_result

    Parameters
    ----------
    FObject: Function Object
    
    idx: int
        dataset index to get chi-squared value from

    scale: float
        Scale to determine area around parameter to inspect
    
    """    
    fig, ax = plt.subplots(1,len(FObject.fit_results[idx].params),figsize = (12,4))
    ax = ax.ravel()
    
    for axi,par in enumerate(FObject.fit_results[idx].params):
        
        params = FObject.fit_results[idx].params.copy()
        scale_par = scale * params[par].value
        
        parfitval = params[par].value
        low = parfitval - scale_par
        high = parfitval + scale_par
        search_line = np.linspace(low, high,50)
        
        chi2 = []
        for x,a in enumerate(search_line):
            params[par].set(a)
            chi2.append(np.sum(np.square(FObject.mod.eval(params,x=FObject.x) - FObject.data[idx])))
        
        ax[axi].plot(search_line,chi2)
        ax[axi].plot(parfitval,chi2[np.argmin(chi2)],'*',markersize = 15)
        ax[axi].axvline(search_line[np.argmin(chi2)],ls='--')
        ax[axi].set_xlabel(par)
        if axi == 0:
            ax[axi].set_ylabel('$\chi^2$')
        for item in ([ax[axi].title, ax[axi].xaxis.label, ax[axi].yaxis.label] +
            ax[axi].get_xticklabels() + ax[axi].get_yticklabels()):
            item.set_fontsize(15)
        
    fig.legend(['$\chi^2$','Fit Value','Lowest $\chi^2$'],bbox_to_anchor=(0.85, 0.4, 0.5, 0.5), loc='lower center',fontsize = 20)
    fig.tight_layout()


def verify_cov_err(FObject,par,true_val,low_bnd = 0.65,hi_bnd = 0.71):
    """
    Verify the confidence intervals estimated from the covariance matrix

    Parameters
    ----------
    FObject: Function Object
    
    par: str
        Parameter name

    true_val: float
        True value of the parameter

    low_bnd: 0 < float < 1
        lower bound on fraction of times true value falls within confidence interval

    hi_bnd: 0 < float < 1
        upper bound on fraction of times true value falls within confidence interval    
    """  
    n = len(FObject.fit_results)

    val = np.array([FObject.fit_results[i].params[par].value for i in range(n)])
    upper = np.array([val[i] + FObject.fit_results[i].params[par].stderr for i in range(n)])
    lower = np.array([val[i] - FObject.fit_results[i].params[par].stderr for i in range(n)])

    # fraction = np.ones_like(lower)[(lower > true_val) | (upper < true_val)].sum()/n
    fraction = np.ones_like(val)[(lower < true_val) & (upper > true_val)].sum()/n

    if low_bnd < fraction < hi_bnd:
        print('Good')
    else:
        print('Bad')

    print(f'True 1 sigma: 0.683\n','Fraction:', fraction)
    fig,ax = plt.subplots(figsize = (15,4))
    p1 = ax.fill_between(np.arange(n), lower, upper)
    p2 = ax.plot(np.arange(n),val,'ko',markersize = 2)

    not_in_ci = np.arange(n)[(lower > true_val) | (upper < true_val)]
    p3 = ax.plot(not_in_ci,true_val*np.ones_like(not_in_ci),'r.',markersize = 3)

    ax.legend(['Fit Value','True Value when not in CI','Confidence Interval'],bbox_to_anchor=(0.7, 0.6, 0.5, 0.5), loc='lower center',fontsize = 20)

    ax.set_xlabel('dataset #')
    ax.set_title(par+' Confidence Interval Results',fontsize = 20)
    for item in ([ax.xaxis.label, ax.yaxis.label] +
        ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

def verify_ci_out(FObject,par,sigma,true_par_val):
    """
    Verify the confidence intervals estimated using the F-test

    Parameters
    ----------
    FObject: Function Object
    
    par: str
        Parameter name

    sigma: int (1,2 or 3)
        n-sigma value to estimate confidence interval from

    true_par_val: float
        True parameter value 
    """  
    if sigma == 1:
        idx_l = 2
        ids_u = 4
    elif sigma ==2:
        idx_l = 1
        ids_u = 5
    elif sigma ==3:
        idx_l = 0
        ids_u = 6
    else:
        raise ValueError('Sigma must be 1,2 or 3') 
        
    verify = []
    for i in range(len(FObject.fit_results)):
        val = FObject.fit_results[i].ci_out[par][3][1]
        upper = FObject.fit_results[i].ci_out[par][ids_u][1]
        lower = FObject.fit_results[i].ci_out[par][idx_l][1]

        verify.append(int(lower < true_par_val < upper))

    percent = sum(verify)/len(verify)
    print(f'True {sigma} sigma',FObject.fit_results[i].ci_out[par][ids_u][0],'\n',f'Calculated {sigma} sigma', percent)
    
    return verify