import lmfit
import numpy as np
import matplotlib.pyplot as plt


class Function:
    """
    Class for holding and fitting data to a function. 
    """
    def __init__(self,x,data,func = None,lmfitmod = None,lmfitpars = None,make_params = True):
        """
        Load the data and function

        Parameters
        ----------
        x: array
            array of x-values of the data

        data: array
            array of data

        func: function
            function to be turned into lmfit Model

        lmfitmod: lmfit Model
            Option to pass an lmfit Model instance

        lmfitpars: lmfit Parameters
            Option to pass an lmfit Parameters instance
        """       
        self.x = x

        if data.ndim == 1:
            self.data = [data]
        else:
            self.data = data

        self.ndatasets = len(self.data)
        

        if (func != None) and (lmfitmod != None):
            raise AttributeError('Must input either a function or an lmfit Model not both')
        if lmfitmod != None:
            self.mod = lmfitmod
        elif func !=None:
            self.mod = lmfit.Model(func)

        if lmfitpars != None:
            self.params = lmfitpars
        
        elif lmfitpars is None:
            if make_params:
                self.params = self.mod.make_params()

        
        
    def fit(self,fit_method = 'leastsq',ci = False,sigma = 2,**kws):
        """
        Fit the data

        Parameters
        ----------
        fit_method: str
            method to be passed to lmfit.Model.fit()

        ci: bool
            evaluate the confidence interval using eval_confidence() method of ModelResult class

        sigma: int (1,2 or 3)
            # of sigma values to evaluate confidence interval over

        """

        self.fit_results = [[] for i in range(len(self.data))]

        for i in range(len(self.data)):
            self.fit_results[i]  = self.mod.fit(self.data[i], self.params, x=self.x, method = fit_method,**kws)  

            if ci == True:
                self.fit_results[i].conf_interval()
        
    
    def plot_fit(self,idx = 0,plot_uncertainty = True, sigma = 2,print_ci = True,**kws):
        """
        Plot a fit_result

        Parameters
        ----------
        idx: int
            index of the fit_result to plot

        ci: bool
            Option to plot the confidence interval report held by ci_out

        sigma: int (1,2 or 3)
            # of sigma values to evaluate confidence interval over

        """  
        print(lmfit.fit_report(self.fit_results[idx].params))
        fig,gs = self.fit_results[idx].plot()

        if plot_uncertainty == True:
            try:
                ax = fig.get_axes()[1]
                dely = self.fit_results[idx].eval_uncertainty(sigma=sigma)
                ax.fill_between(self.x, self.fit_results[idx].best_fit-dely,
                    self.fit_results[idx].best_fit+dely, color='#888888')
            except:
                print('Could not evaluate uncertainty')
                pass


        if print_ci == True and self.fit_results[idx].ci_out is not None:
            print('='*50)
            lmfit.printfuncs.report_ci(self.fit_results[idx].ci_out)

        return fig, ax
            
            

        
def lorentzian_func(x,A,C,W):

    return A / (1+( (x-C)/(W/2))**2 )        
        
class Lorentzian(Function):
    """A model based on a Lorentzianfunction with three Parameters:
    'A' is the amplitude 
    'C' is the center position
    'W' is the full width at half maximum
    'A' 
    .. math::

        $f(x; A, C, W) = \frac{A}{1 + ( (x - C)/(w/2) )^2}}$

    """
    def __init__(self,x,data):

        super().__init__(x,data,func = lorentzian_func,make_params = False)

        self.params = self.guess_from_data()

    def guess_from_data(self,idx = 0):
        """
        guess the paramters from the data

        Parameters
        ----------
        idx: int
            index of the data to guess peak from

        """     
        maxval = self.data[idx].max()
        halfmax = maxval / 2
        maxpos = self.data[idx].argmax()
        leftpos = (np.abs(self.data[idx][:maxpos] - halfmax)).argmin()
        rightpos = (np.abs(self.data[idx][maxpos:] - halfmax)).argmin() + maxpos
        fwhm = self.x[rightpos] - self.x[leftpos]   

        pars = self.mod.make_params(A=maxval, C=self.x[maxpos], W=fwhm)
        pars['W'].set(min=0.0)
        return pars 