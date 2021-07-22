import lmfit
import numpy as np
import matplotlib.pyplot as plt

class OptimizationAlgorithms:

    """Class for demonstration of different algorithms making up the Levenberg-Marquardt algorithm"""

    def __init__(self,func,x,true_params):
        """

        Parameters
        ----------
        func: function
            function to apply optimization algorithms on

        x: array
            1-d array to search parameter space through. The true parameters must both be in this range

        true_params: array
            1-d array of true parameters

        """   

        self.f = func
        self.x = x
        self.beta_true = true_params
        self.y_true = self.f(self.x,self.beta_true)

        self.build_parspace_grid()

    def J(self,beta, dx=1e-8):
        """calculate the jacobian at parameter values beta"""

        n = len(beta)
        m = len(self.x)
        func = self.f(self.x,beta)
        jac = np.zeros((m, n))
        for j in range(n):  # through columns to allow for vector addition
            Dxj = (abs(beta[j])*dx if beta[j] != 0 else dx)
            beta_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(beta)]
            jac[:, j] = (self.f(self.x,beta_plus) - func)/Dxj
            
        return jac


    def build_parspace_grid(self):
        """ Build the matrices used to visualize and optimize algorithms
        
        axis-0: beta[1] values
        axis-1: beta[0] values
        axis-2 : x-datapoints
        
        """

        self.aa,self.bb = np.meshgrid(self.x, self.x)
        aaa = np.repeat(self.aa[:,:,np.newaxis],len(self.x),axis=2)
        bbb = np.repeat(self.bb[:,:,np.newaxis],len(self.x),axis=2)


        par_space = self.f(self.x,[aaa,bbb])
        yyy = self.f(self.x,self.beta_true)*np.ones_like(par_space)

        self.squared_resid_2d = ((yyy - par_space)**2).sum(axis = 2)


    def gradient_descent(self,beta0,dp,niters = 30,tol = 0.1):
        
        """
        Perform gradient descent until chi-squared reaches some tolerance or the number of iterations is reached.

        Parameters
        ----------
        beta0: 1-d array
            starting position of parameters

        dp: float
            step for gradient descent

        niters: int
            number of iterations of algorithm
        
        tol:
            chi-squared tolerance threshold

        """
        beta = [beta0]
        chi2 = []
        h_n = []

        for i in range(niters):
            
            chi2.append(np.sum((self.y_true - self.f(self.x,beta[i]))**2))
            if chi2[i] < tol:
                break
                
            Jac = self.J(beta[i])
            h = np.dot(Jac.T,(self.y_true - self.f(self.x,beta[i]))[:,np.newaxis])
            
            h_n.append(h/np.sqrt(np.sum(h**2)))


            beta.append(beta[i] + h_n[i][:,0]*dp)
            
            
        fig, ax = plt.subplots(figsize = (10,10))
        ax.contour(self.aa, self.bb, self.squared_resid_2d,vmin = 0,levels = 200)
        ax.plot(beta[0][0],beta[0][1],'ro')

        ax.plot([beta[i][0] for i in range(1,len(beta))],[beta[i][1]for i in range(1,len(beta))],'ko-')

        ax.set_xlabel('beta[0]')
        ax.set_ylabel('beta[1]')
        
        for item in ([ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)


    def gauss_newton(self,beta0,niters = 30,tol = 0.1):
        """
        Perform Gauss-Newton algorithm until chi-squared reaches some tolerance or the number of iterations is reached.

        Parameters
        ----------
        beta0: 1-d array
            starting position of parameters

        niters: int
            number of iterations of algorithm
        
        tol:
            chi-squared tolerance threshold

        """
        beta = [beta0]
        chi2 = []
        h = []

        for i in range(niters):
            
            y_beta = self.f(self.x,beta[i])
            
            chi2.append(np.sum((self.y_true - self.f(self.x,beta[i]))**2))
            if chi2[i] < tol:
                break
                
            Jac = self.J(beta[i])
            J2=np.dot(np.linalg.inv(np.dot(Jac.T,Jac)),Jac.T) #Calculate J2=(J^T.J)^-1.J^T
            
            h.append(np.dot(J2,(self.y_true - y_beta)[:,np.newaxis]))
            
            beta.append(beta[i] + h[i].reshape(2,))
            
        fig, ax = plt.subplots(figsize = (10,10))
        ax.contour(self.aa, self.bb, self.squared_resid_2d,vmin = 0,levels = 200)
        ax.plot(beta[0][0],beta[0][1],'ro')

        ax.plot([beta[i][0] for i in range(1,len(beta))],[beta[i][1]for i in range(1,len(beta))],'ko-')
            
        ax.set_xlabel('beta[0]')
        ax.set_ylabel('beta[1]')
        
        for item in ([ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)


    def levenberg_marquardt(self,beta0,lamb,niters = 30,tol = 0.1):
        """
        Perform Levenberk-Marquardt algorithm until chi-squared reaches some tolerance or the number of iterations is reached.

        Parameters
        ----------
        beta0: 1-d array
            starting position of parameters
        
        lamb: float
            starting lambda value

        niters: int
            number of iterations of algorithm
        
        tol:
            chi-squared tolerance threshold

        """
        lamb = [lamb]

        beta = [beta0]
        chi2 = []
        parshape = len(beta0)
        nu = len(self.x)


        for i in range(niters):

            y_beta = self.f(self.x,beta[i])
            
            chi2.append(np.sum((self.y_true - self.f(self.x,beta[i]))**2))
            if chi2[i] < tol:
                break
            
            Jac = self.J(beta[i])
            
            lambscale = lamb[i]*np.eye(parshape)*np.diag(np.dot(Jac.T,Jac))
            J2=np.dot(np.linalg.inv(np.dot(Jac.T,Jac) + lambscale),Jac.T) #Calculate J2=(J^T.J+lamb.I)^-1.J^T
                
            h = np.dot(J2,(self.y_true - y_beta)[:,np.newaxis])

            beta_new = beta[i] + h.reshape(2,)
            
            chi_new = np.sum((self.y_true - self.f(self.x,beta_new))**2)

            if chi_new > chi2[i]:
                lambnew = np.min([1e7,lamb[i]*nu])
                lamb.append(lambnew)
            elif chi_new < chi2[i]:
                lambnew = np.max([1e-7,lamb[i]/nu])
                lamb.append(lambnew)
            
            beta.append(beta_new)
            
        fig, (ax1,ax2) = plt.subplots(1,2,figsize = (20,10))
        ax1.contour(self.aa, self.bb, self.squared_resid_2d,vmin = 0,levels = 200)
        ax1.plot(beta[0][0],beta[0][1],'ro')

        ax1.plot([beta[i][0] for i in range(1,len(beta))],[beta[i][1]for i in range(1,len(beta))],'ko-')


        ax1.set_xlabel('beta[0]')
        ax1.set_ylabel('beta[1]')

        ax2.plot(lamb,'-o')

        ax2.set_xlabel('Iteration #')
        ax2.set_ylabel('$\lambda$')

        for ax in [ax1,ax2]:
            for item in ([ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(15)