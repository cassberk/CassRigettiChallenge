from ipywidgets.widgets import Label, FloatProgress, FloatSlider, Button, Checkbox,FloatRangeSlider, Button, Text,FloatText,\
Dropdown,SelectMultiple, Layout, HBox, VBox, interactive, interact, Output,jslink
from IPython.display import display, clear_output
from ipywidgets import GridspecLayout
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
import ipywidgets

import numpy as np
from copy import deepcopy as dc
import matplotlib.pyplot as plt
import lmfit as lm
from lmfit.model import load_model, load_modelresult

import sys
import os



class ParameterWidgetGroup:
    """Modified from existing lmfit.ui.ipy_fitter"""
    def __init__(self, par, slider_ctrl=True,sliderlims = None):
        self.par = par
        self.slider_ctrl = slider_ctrl
        self.sliderlims = sliderlims

        widgetlayout = {'flex': '1 1 auto', 'width': 'auto', 'margin': '0px 0px 0px 0px'}
        width = {'description_width': '10px'}


        # Define widgets.
        self.value_text = FloatText(
            value=np.round(self.par.value,2),
            placeholder='Value',
            disabled=False,
            layout = Layout(width = '200px', margin = '0 5px 0 0')
            )
        self.expr_text = Text(
            value=self.par.expr,
            placeholder='Choose Your Destiny',
            disabled=False,
            layout = Layout(width = '200px', margin = '0 5px 0 0')
            )
        self.min_text = FloatText(
            value=np.round(self.par.min,2),
            placeholder='min',
            disabled=False,
            layout = Layout(width = '100px', margin = '0 0 15px 0')
            )
        self.max_text = FloatText(
            value=np.round(self.par.max,2),
            placeholder='min',
            disabled=False,
            layout = Layout(width = '100px', margin = '0 0 15px 0')
            )
        self.min_checkbox = Checkbox(description='min', style=width, layout=widgetlayout)
        self.max_checkbox = Checkbox(description='max', style=width, layout=widgetlayout)
        self.vary_checkbox = Checkbox(value=bool(self.par.vary),
            description=self.par.name,
            disabled=False,
            indent=False,
            layout = Layout(width = '200px', margin = '0 5px 0 0')
            )
        if self.slider_ctrl is True:
            # if self.par.expr is None:
            #     dis = False
            # else:
            #     dis = True
            self.ctrl_slider = FloatSlider (
                        value=self.par.value,
                        min = self.sliderlims[0],
                        max = self.sliderlims[1], ### Need to figure out a way to set this
                        step  = 0.01,
                        # disabled=dis,
                        description = self.par.name,
                        # style = {'description_width': 'initial','handle_color' : element_color['_'.join(self.par.name.split('_')[:-1]+[''])]},
                        layout = Layout(width = '350px', margin = '0 0 5ps 0')
                        )

            widget_link =jslink((self.ctrl_slider, 'value'), (self.value_text, 'value'))
        # else:
        #     self.ctrl_slider = None


        # Set widget values and visibility.
        if self.par.value is not None:
            self.value_text.value = self.par.value
        min_unset = self.par.min is None or self.par.min == -np.inf
        max_unset = self.par.max is None or self.par.max == np.inf
        self.min_checkbox.value = not min_unset
        self.min_text.value = self.par.min
        # self.min_text.disabled = min_unset
        self.max_checkbox.value = not max_unset
        self.max_text.value = self.par.max
        # self.max_text.disabled = max_unset
        self.vary_checkbox.value = bool(self.par.vary)

        # Configure widgets to sync with par attributes.
        self.value_text.observe(self._on_value_change, names='value')
        self.expr_text.observe(self._on_expr_change, names='value')

        self.min_text.observe(self._on_min_value_change, names='value')
        self.max_text.observe(self._on_max_value_change, names='value')
        # self.min_checkbox.observe(self._on_min_checkbox_change, names='value')
        # self.max_checkbox.observe(self._on_max_checkbox_change, names='value')
        self.vary_checkbox.observe(self._on_vary_change, names='value')

    def _on_value_change(self, change):
        self.par.set(value=change['new'])

    def _on_expr_change(self, change):
        self.par.set(expr=change['new'])

    def _on_min_checkbox_change(self, change):
        self.min_text.disabled = not change['new']
        if not change['new']:
            self.min_text.value = -np.inf

    def _on_max_checkbox_change(self, change):
        self.max_text.disabled = not change['new']
        if not change['new']:
            self.max_text.value = np.inf

    def _on_min_value_change(self, change):
        if not self.min_checkbox.disabled:
            self.par.set(min=change['new'])

    def _on_max_value_change(self, change):
        if not self.max_checkbox.disabled:
            self.par.set(max=change['new'])

    def _on_vary_change(self, change):
        self.par.set(vary=change['new'])

    def close(self):
        # one convenience method to close (i.e., hide and disconnect) all
        # widgets in this group
        self.value_text.close()
        self.expr_text.close()
        self.min_text.close()
        self.max_text.close()
        self.vary_checkbox.close()
        self.min_checkbox.close()
        self.max_checkbox.close()

    def get_widget(self):
        box = VBox([self.vary_checkbox, 
                            self.value_text, 
                            self.expr_text, 
                            HBox([self.min_text, self.max_text]),
                            ])
        return box

    def update_widget_group(self,par):
        # print(par)
        self.par = par
        self.vary_checkbox.value =  bool(self.par.vary)
        self.min_text.value =  self.par.min
        self.max_text.value =  self.par.max
        # self.define_widgets()
        if par.expr != None:
            self.expr_text.value = self.par.expr
        elif self.par.expr == None:
            self.value_text.value = self.par.value
        

    # Make it easy to set the widget attributes directly.
    @property
    def value(self):
        return self.value_text.value

    @value.setter
    def value(self, value):
        self.value_text.value = value

    @property
    def expr(self):
        return self.expr_text.value

    @value.setter
    def expr(self, expr):
        self.expr_text.value = expr

    @property
    def vary(self):
        return self.vary_checkbox.value

    @vary.setter
    def vary(self, value):
        self.vary_checkbox.value = value

    @property
    def min(self):
        return self.min_text.value

    @min.setter
    def min(self, value):
        self.min_text.value = value

    @property
    def max(self):
        return self.max_text.value

    @max.setter
    def max(self, value):
        self.max_text.value = value

    @property
    def name(self):
        return self.par.name



class fitting_panel:
    """Class for building the interactive fitting part of the gui"""

    def __init__(self, fit_object, n_scans):
        """Parameters
        ----------
        fit_object: Function object instance
            Function object is passed to the fitting panel. This then controls the fit and plot methods 
            of those classes.
        n_scans: int
            number of datasets held by fit object


        Returns
        -------


        See Also
        --------
        :func:
        """

        self.fit_object = fit_object
       
 

        #Fitting functions supported by lmfit
        fitting_options = [('Levenberg-Marquardt', 'leastsq'),\
        ('Least-Squares minimization Trust Region Reflective method ',' least_squares'),\
        ('differential evolution','differential_evolution'),\
        ('brute force method', 'brute'),\
        ('basinhopping', 'basinhopping'),\
        ('Adaptive Memory Programming for Global Optimization','ampgo'),\
        ('Nelder-Mead','nelder'),\
        ('L-BFGS-B','lbfgsb'),\
        ('Powell','powell'),\
        ('Conjugate-Gradient','cg'),\
        ('Newton-CG','newton'),\
        ('Cobyla','cobyla'),\
        ('BFGS','bfgs'),\
        ('Truncated Newton','tnc'),\
        ('Newton-CG trust-region','trust-ncg'),\
        ('nearly exact trust-region','trust-exact'),\
        ('Newton GLTR trust-region','trust-krylov'),\
        ('trust-region for constrained optimization','trust-constr'),\
        ('Dog-leg trust-region','dogleg'),\
        ('Sequential Linear Squares Programming','slsqp'),\
        ('Maximum likelihood via Monte-Carlo Markov Chain','emcee'),\
        ('Simplicial Homology Global Optimization','shgo'),\
        ('Dual Annealing optimization','dual_annealing')]


        #Define Widgets
        self.fit_button = Button(description="Fit")
        self.plot_button = Button(description="Plot")   
        self.save_fig_button = Button(description="Save Figure") 
        self.autofit_button = Button(description="View Autofit")
        
        self.fit_method_widget = Dropdown(
            options = fitting_options,
            value='leastsq',
            description = 'Fit Method',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '400px', margin = '0 0 5ps 0')
            )
          
            
        self.data_to_fit_widget = SelectMultiple(
            options=[None, 'All'] + list(np.arange(0,n_scans)), # Get rid ofthis once data dict is no longer used
            value = ('All',),
            description='Data to fit',
            style = {'description_width': 'initial'},
            disabled=False
            )
        
        self.plot_all_chkbx = Checkbox(
            value= False,
            description='Plot all fit_results',
            style = {'description_width': 'initial'},
            disabled=False,
            indent=False
            )
        

        self.use_prev_fit_result_params = Checkbox(
            value= False,
            description='Update Params with Previous Fit Result',
            style = {'description_width': 'initial'},
            disabled=False,
            indent=False
            )     
        
 
        self.save_fig_name = Text(
            description = 'Save figure name',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '400px', margin = '0 0 5ps 0')
            )


        self.sigma_ci_widget =  Dropdown(
            options=[int(i) for i in range(1,4)],      
            value = 1,
            description='Sigma Value for Confidence Inverval',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '300px', margin = '0 0 5ps 0')
            )

        self.calculate_F_test_CI_chkbx= Checkbox(
            value= False,
            description='Calculate Confidence Intervals with F-test',
            style = {'description_width': 'initial'},
            disabled=False,
            indent=False
            )  

        self.autofit_chkbx= Checkbox(
            value= False,
            description='autofit',
            style = {'description_width': 'initial'},
            disabled=False,
            indent=False
            )     

        self.plotfit_chkbx= Checkbox(
            value= True,
            description='plot fits',
            style = {'description_width': 'initial'},
            disabled=False,
            indent=False
            )      


        ### Build the Fitting Panel
        v1 = VBox([self.fit_method_widget])
        h1 = HBox([v1,self.data_to_fit_widget]) 
        
        fig_save = HBox([self.save_fig_name,self.save_fig_button])
        
        h2 = HBox([self.fit_button,self.plot_button,fig_save])
        
        vfinal = VBox([h1, h2, self.plotfit_chkbx,self.sigma_ci_widget,self.calculate_F_test_CI_chkbx, HBox([self.autofit_chkbx,self.autofit_button]), self.use_prev_fit_result_params, \
                               self.plot_all_chkbx])
        
        display(vfinal)
        
        out = Output()
        display(out)


        #Define button functions
        @self.save_fig_button.on_click
        def save_fig_on_click(b):
            with out:
                if not hasattr(self,'fig'):
                    print('There is no figure object!')
                    return

                if not os.path.exists(os.path.join(os.getcwd(),'figures')):
                    os.makedirs(os.path.join(os.getcwd(),'figures'))
                    if not os.path.exists(os.path.join(os.getcwd(),'figures','fits')):
                        os.makedirs(os.path.join(os.getcwd(),'figures','fits'))

                save_location = os.path.join(os.getcwd(),'figures','fits',self.save_fig_name.value)          

                self.fig.savefig(save_location, bbox_inches='tight')

        @self.fit_button.on_click
        def fit_on_click(b):
            self.BE_adjust = 0  #change this once the new xps package is ready
            with out:
                # if hasattr(self,"fig"):
                clear_output(True)
                
                self.fit_data()
                show_inline_matplotlib_plots()
        
        
        
        @self.plot_button.on_click
        def plot_on_click(b):
            with out:
                clear_output(True)

                self.plot_data()
                show_inline_matplotlib_plots()    

                
        @self.autofit_button.on_click
        def plot_on_click(b):
            with out:
                if self.data_to_fit_widget.value == 'All':
                    specnum = 0
                else:
                    specnum = self.data_to_fit_widget.value
                print(specnum)
                if not hasattr(self,'autofit'):
                    # self.autofit = XPyS.autofit.autofit.autofit(self.fit_object.x,self.fit_object.data[specnum[0]],self.fit_object.orbital)
                    self.autofit = self.get_autofit_model()
                # elif hasattr(self,'autofit'):
                    # self.autofit.guess_params(energy = self.fit_object.x,intensity = self.fit_object.data[specnum[0]])
                print(self.fit_object.data[specnum[0]])
                self.autofit.guess_params(self.fit_object.data[specnum[0]],self.fit_object.x)

                for par in self.autofit.guess_pars.keys():
                    self.fit_object.params[par].value = self.autofit.guess_pars[par]



    # Fitting Panel Methods            
    def fit_data(self):
        """Function called when self.fit_button is clicked. This will call the fit method on the Function object instance provided certain
        conditionals are met.


        See Also
        --------

        :func:Function.fit
        """

        ### Fitting Conditionals
        if self.fit_method_widget.value ==None:
            print('Enter a fitting Method',flush =True)
            return     
        
        if 'All' in self.data_to_fit_widget.value:
            print('%%%% Fitting all datsets... %%%%',flush =True)          
            fit_points = None
            
        elif self.data_to_fit_widget.value[0] == None:
#             self.fit_iter_idx = range(0)
            print('No Specta are selected!',flush =True)
            return
        else:
            fit_points = list(self.data_to_fit_widget.value)
            print('%%%% Fitting dataset ' + str(fit_points)+'... %%%%',flush =True) 

        # self.fit_object.fit(specific_points = fit_points,plotflag = False, track = False, update_with_prev_pars = self.use_prev_fit_result_params.value,\
        #     autofit = self.autofit_chkbx.value)

        self.fit_object.fit(fit_method = self.fit_method_widget.value,ci = self.calculate_F_test_CI_chkbx.value,sigma = 2)
        self.plot_data()


    def plot_data(self):
        """Function called when self.plot_button is clicked. This will call the plot_fitresults() method on the function object instance provided certain
        conditionals are met.


        See Also
        --------

        :func:function.plot_fit()
        """

        ### Plotting Conditionals
        if self.plot_all_chkbx.value is True:
            print('Plotting all datasets ...')
            plot_points = [j for j,x in enumerate(self.fit_object.fit_results) if x]

        elif self.plot_all_chkbx.value is False:
            print('Plotting' + str(self.data_to_fit_widget.value) + ' dataset ...')

            if self.data_to_fit_widget.value[0] == None:
                print('Error!: You are trying to Not plot all results and Not fit any dataset')
                return

            elif 'All' in self.data_to_fit_widget.value:
                plot_points = [j for j,x in enumerate(self.fit_object.fit_results) if x]

            else:
                plot_points = dc(list(self.data_to_fit_widget.value))  
        for i in plot_points:
            print('='*100)
            print('Dataset: ',i)
            self.fig, self.axs = self.fit_object.plot_fit(idx = i,print_ci = self.calculate_F_test_CI_chkbx.value,sigma = self.sigma_ci_widget.value) 
            plt.show()

    




class interactive_fit:
    """Main class for performing interactive fitting of xps signal."""

    def __init__(self,function_object):
        """Parameters
        ----------
        input_object: XPyS.sample.sample instance, list, dict
            The input object can be a few different things
            1. An sample object
            2. A list of sample objects. If this is the case then the sample_name attribute wil be used to select the samples
            3. A dictionary of sample objects where the key is the sample identifier. That sample identifier will be used to 
            select a given sample


        Returns
        -------


        See Also
        --------
        :func:

        """
        self.function_object = function_object
        self.create_full_panel(self.function_object)


    def create_full_panel(self,function_object,control_parameters = None, control_limits = None):
        """Function to build the full interactive fitting gui. This will call functions to make the parameter widgets, the 
        interactive plot, and the panel for controlling the fit


        Parameters
        ----------
        function_object: FunctionFitting.Function instance
            All of the adjustable parameters will be built based of the model used to fit the dataset. 


        Returns
        -------


        See Also
        --------
        :func:  self.make_parameter_panel(), self.make_interactive_plot(), self.fitting_panel

        """

        self.function_object = function_object

        self.x= function_object.x
        self.data = function_object.data

        self.prefixlist = [comp.prefix for comp in self.function_object.mod.components]

        
        # Make a list of all the relevant parameters. lmfit models can have parameters that depend on other parameters
        # and we are not interested in them. Next, make a list of the prefixes that we want a control bar for. This is a remnant 
        # of me using element_ctrl as a list of integers specifying which prefixes in pairlist* I want to control. 
        # element_ctrl could probably be replaced by a dict eventually. Last, create a dictionary of bool values for each parameter
        # to pass to ParameterWidgetGroup telling it to make a control slider or not. 

        # *pairlist is used to link different peaks together in an xps doublet, since you have to make different lmfit models for each peak
        

        self.rel_pars = [par for component_pars in [model_component._param_names for model_component in self.function_object.mod.components] \
            for par in component_pars]
        

        # In case there are paramters that depend on one another we can specify which ones we want control sliders for.
        # This can be automated by using prefixes. It is not functional yet...
        if control_parameters is None:
            self.ctrl_pars = {par: True for par in self.rel_pars}
        else:
            self.ctrl_prefixes = [[prefix for pairs in self.function_object.pairlist \
                for prefix in pairs][i] for i in self.function_object.element_ctrl]

        
            self.ctrl_pars = {par: any(x in par for x in self.ctrl_prefixes) for par in self.rel_pars}

        if control_limits is None:
            self.ctrl_lims = {}

            for par in self.rel_pars:
                self.ctrl_lims[par] = (0,5*self.function_object.params[par].value)
        else:
            if set(self.rel_pars) == set(list(control_limits.keys())):
                self.ctrl_lims = control_limits
            else:
                raise KeyError('The control_limit keys do not match the rel_pars attribute')

        

        self.make_parameter_panel(parameters = self.function_object.params)
        self.make_interactive_plot()
        self.fitting_panel = fitting_panel(self.function_object,n_scans = len(self.function_object.data))



    def make_parameter_panel(self,parameters = None):
        """Function to build the parameter panel for interactively controling which parameters are varied as well as parameter ranges
        and parameter expressions (making one parameter depend on another). A ParameterWidgetGroup instance will be created for each
        parameter and stored in the paramwidgets dictionary.


        Parameters
        ----------
        parameters: lmfit parameters object instance


        Returns
        -------


        See Also
        --------
        :class:  ParameterWidgetGroup

        """

        if parameters == None:
            print('No parameters specified')
            return
        box_layout = Layout(display='flex',
                                    flex_flow='column',
                                    align_items='stretch',
                                    width='100%')
        self.paramwidgetscontainer = VBox([], layout=box_layout)

                
        self.paramwidgets = {p_name:ParameterWidgetGroup(p,slider_ctrl = self.ctrl_pars[p_name],sliderlims = self.ctrl_lims[p_name])\
                for p_name, p in parameters.items() if p_name in self.rel_pars}

        ### The children are the paramwidgets for each model
        self.paramwidgetscontainer.children = [HBox([self.paramwidgets[comp_name].get_widget() \
            for comp_name in self.function_object.mod.components[i]._param_names]) for i in range(len(self.function_object.mod.components))]
        display(self.paramwidgetscontainer)



    def make_interactive_plot(self):
        """Function to build the plotting part GUI. This is made up of the matplotlib figure to view modifications to the parameters.
        It is also made up of the sliders held by the ParameterWidgetGroup for the element_ctrl parameters. It also has the widget to
        change which dataset is being fit. As well as the widgets to reset the sliders if a limit of the range is reached.
        As well as widgets to change the parameters to fit_result parameters.


        Parameters
        ----------


        Returns
        -------


        See Also
        --------
        :class:  ParameterWidgetGroup
        :func: interactive_plot()


        """
        self.change_pars_to_fit_button = Button(
            description="Change Parameters to Fit Result",
            layout = Layout(width = '300px', margin = '0 0 5ps 0')
            ) 

        self.reset_slider_lims_button = Button(
            description="Reset Slider Max",
            layout = Layout(width = '300px', margin = '0 0 5ps 0')
            ) 

        self.data_init_widget =  Dropdown(
            options=list(np.arange(0,len(self.data))),      
            value = 0,
            description='Dataset',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '200px', margin = '0 0 5ps 0')
            )

        self.reset_slider_widget =  Dropdown(
            options= [par for par in self.ctrl_pars.keys() if self.ctrl_pars[par]],  
            value = None,
            description='Slider Reset',
            style = {'description_width': 'initial'},
            disabled=False,
            layout = Layout(width = '200px', margin = '0 0 5ps 0')
            )        

        self.wlim = FloatRangeSlider (
                value=[np.min(self.x), np.max(self.x)],
                min = np.min(self.x),
                max =np.max(self.x),
                step  = 0.01,
                description = 'xlim',
                style = {'description_width': 'initial'},
                layout = Layout(width = '300px', margin = '0 0 5ps 0')
                )      
    

        out = Output()
        display(out)

        @self.change_pars_to_fit_button.on_click
        def plot_on_click(b):
            with out:

                self.function_object.params = self.function_object.fit_results[self.data_init_widget.value].params.copy()
                for pars in self.paramwidgets.keys():
                    self.paramwidgets[pars].update_widget_group(self.function_object.params[pars])
  

        @self.reset_slider_lims_button.on_click
        def plot_on_click(b):
            with out:
                if self.reset_slider_widget.value !=None:
                    self.paramwidgets[self.reset_slider_widget.value].ctrl_slider.max  = 2*self.paramwidgets[self.reset_slider_widget.value].ctrl_slider.value

        # Create the interactive plot, then build the slider/graph parameter controls
        plotkwargs = {**{pw.name:pw.ctrl_slider for pw in self.paramwidgets.values() if hasattr(pw,'ctrl_slider')},\
            **{'xlim':self.wlim}}
        self.intplot = interactive(self.interactive_plot,**plotkwargs)
        
        vb = VBox(self.intplot.children[0:-1])
        vb2 = VBox([HBox([VBox([self.data_init_widget,self.reset_slider_widget]),VBox([self.change_pars_to_fit_button,self.reset_slider_lims_button])]),self.intplot.children[-1]])
        hb = HBox([vb,vb2])
            
        display(hb)
        

    def interactive_plot(self,*args,**kwargs):
        """interactive plotting function to be called by ipywidget.interactive"""        

        fig,ax = plt.subplots(figsize=(8,6))
        p1 = ax.plot(self.x,self.data[self.data_init_widget.value],'bo')
        p2 = ax.plot(self.x,self.function_object.mod.eval(self.function_object.params,x=self.x) , color = 'black')             
                                                
            
        ax.set_xlim(np.min(kwargs['xlim']),np.max(kwargs['xlim']))
        
        plt.show()

