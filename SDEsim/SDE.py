import torch
from collections import defaultdict
import time
from pathlib import Path
import os
import inspect
import pickle

def histc(y, bins:int, max, min):
    #for future.
    if len(y.shape)==1:
        return torch.histc(y, bins=bins, max=max, min=min)
    else:
        return torch.stack([torch.histc(y[i], bins=bins, max=max[i], min=min[i]) for i in range(y.shape[0])])

class SDE:

    """A class for simulating Autonomous SDEs using the Ito Scheme.
        The following arguments are possilbe:
        > num_par: int/long, number of parallel simulations to run.
        > method: string, available methods: 'euler_maruyama', 'milstein' (default), 'taylor_15'
        > dtype: string, available dtypes: 'float32', 'float64' (default)
        > device: string, available devices: 'cpu', 'cuda' (default)
        > adaptive_stepping: bool, True (default) turns on adaptive time steps.
        > tolerance: float, local error tolerance for adaptive stepping. Default value = 1e-3
        > dt_init: float, initial step size. Deafult value = 1e-10
        """

    def __init__(self, num_par : int, **kwargs):
   
        self.num_par = num_par #number of parallel simulations
        
        dtype = kwargs.get('dtype', 'float64')
        if dtype == 'float32':
            self.dtype = torch.float32
        elif dtype == 'float64':
            self.dtype = torch.float64
        else:
            raise Exception("Choose 'float32' or 'float64' as dtype")
        
        device = kwargs.get('device', 'cuda')
        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif device == 'cuda':
                print('Warning: No CUDA device available. Reverting to CPU')
                self.device = torch.device('cpu')
        elif device == 'cpu':
            self.device = torch.device('cpu')
        else:
            raise Exception('Device not recognized')
        
        method = kwargs.get('method', 'milstein')
        if method == 'milstein':
            self._order_of_conv = torch.tensor(1.0, dtype = self.dtype, device = self.device)
        elif method == 'euler_maruyama':
            self._order_of_conv = torch.tensor(0.5, dtype = self.dtype, device = self.device)
        elif method == 'taylor_15':
            self._order_of_conv = torch.tensor(1.5, dtype = self.dtype, device = self.device)
        else:
            raise Exception("Method not supported")
        self.method = method
        
        self.y_init = torch.zeros((num_par,), dtype = self.dtype, device = self.device)
        
        ####Error calculation and Adaptive stepping parameters.
        self.adaptive_stepping = kwargs.get('adaptive_stepping', True) #use adaptive stepping
        self.dt_init = torch.tensor(kwargs.get('dt_init', 1e-10), dtype = self.dtype, device = self.device)#initial step size
        self._tolerance = torch.tensor(kwargs.get('tolerance', 1e-3), dtype = self.dtype, device = self.device) #Maximum allowed local error.
        self._fac = torch.tensor(kwargs.get('fac', 0.9), dtype = self.dtype, device = self.device)
        self._facmin = torch.tensor(kwargs.get('facmin', 0.2), dtype = self.dtype, device = self.device)
        self._facmax = torch.tensor(kwargs.get('facmax', 1.3), dtype = self.dtype, device = self.device)
        
        
        ####Internal variables####################
        self.debug = False
        self.gauss_estimate=False #append gaussian approximation to information rate calculation.
        self._mean_shift_threshold = torch.tensor(0.2, dtype = self.dtype, device = self.device)
        self._functions_compiled = False
        self._err_scale_exp = 1/(1+self._order_of_conv)
        self._velocity_avg_dt = torch.tensor(1e-8, dtype = self.dtype, device = self.device)
        self._shape = (num_par, )
        self.fn_def_strings = ['f', 'g', 'df', 'dg', 'd2f', 'd2g']


        self.set_stats()
    
    @property
    def tolerance(self):
        return self._tolerance
    @property
    def fac(self):
        return self._fac
    @property
    def facmin(self):
        return self._facmin
    @property
    def facmax(self):
        return self._facmax   
    
    
    
    @tolerance.setter
    def tolerance(self, new_val):
        self._tolerance = torch.tensor(new_val, dtype = self.dtype, device = self.device) 
        
        #recompile step error functions.
        if self._functions_compiled == True:
            self.set_functions(*self.fn)
            
    @fac.setter
    def fac(self, new_val):
        self._fac = torch.tensor(new_val, dtype = self.dtype, device = self.device) 
            
    @facmin.setter
    def facmin(self, new_val):
        self._facmin = torch.tensor(new_val, dtype = self.dtype, device = self.device) 
    
    @facmax.setter
    def facmax(self, new_val):
        self._facmax = torch.tensor(new_val, dtype = self.dtype, device = self.device) 
    
    def _velocity(self, y, t):
        dt = self._velocity_avg_dt
        xi = torch.normal(mean=0.0, std=1.0, size=y.shape, device = y.device, dtype=y.dtype)
        return (self._step_forward_uncompiled(y, xi*dt**(-0.5), dt, t)-y)/dt

    @property
    def velocity_avg_dt(self):
        return float(self._velocity_avg_dt)
    
    @velocity_avg_dt.setter
    def velocity_avg_dt(self, new_val):
        self._velocity_avg_dt = torch.tensor(new_val, dtype = self.dtype, device=self.device)
        if self._functions_compiled:
            self.velocity = torch.jit.trace(self._velocity, (self.y_init, self.dt_init), check_trace = False)
    
    @staticmethod
    def _milstein(Y, dW, dt, fn):
        #Following expressions are for multi dimensional case
        if callable(fn[0]): f = fn[0](Y)
        else: f = fn[0]
        if callable(fn[1]): g = fn[1](Y)
        else: g = fn[1]
            
        dg = fn[3]
        
        terms = Y + f*dt + g*dW 
        if dg != None:
            terms += 0.5*g*dg(Y)*(dW**2 - dt)
        return terms
    
    @staticmethod
    def _euler_maruyama(Y, dW, dt, fn):
        #Following expressions are for multi dimensional case
        if callable(fn[0]): f = fn[0](Y)
        else: f = fn[0]
        if callable(fn[1]): g = fn[1](Y)
        else: g = fn[1]
        
        ode_term = Y + f*dt
        stochastic_term = g*dW
        return ode_term + stochastic_term
    
    
    def _taylor_15(self, y, dw, dt, fn):
        ##Section 10.4, Numerical Solution of Stochastic Differential Equations - Peter E. Kloeden Eckhard Platen (1999)
        #Following expressions are for multi dimensional case
        if callable(fn[0]): f = fn[0](y)
        else: f = fn[0]
        if callable(fn[1]): g = fn[1](y)
        else: g = fn[1]
            
        df, dg, d2f, d2g = fn[2], fn[3], fn[4], fn[5]
        
        if df != None or dg != None:
            temp = torch.normal(mean=0.0, std=1.0, size=dw.shape, device = self.device, dtype=self.dtype)/3**0.5
            dz = 0.5*dt**(1.5)*(dw/(dt**0.5) + temp)
        a = f
        b = g
        
        terms = y + a*dt + b*dw
        if df != None:
            a1 = df(y)
            terms += a1*(b*dz + 0.5*a*dt**2)
        if d2f != None:
            a2 = d2f(y)
            terms += 0.25*b**2*a2*dt**2
        if dg != None:
            b1 = dg(y)
            terms += b1*(0.5*b*(dw**2-dt) + a*(dw*dt - dz) + 0.5*b*b1*(dw**2/3 - dt)*dw)
        if d2g != None:
            b2 = d2g(y)
            terms += 0.5*b**2*b2*(dw**2/3 - dz)
        
        return terms
    
    def _local_error(self, y_new, y_old, dW, dt, t=0):
            #Ilie, Silvana, Kenneth R. Jackson, and Wayne H. Enright. 
            #"Adaptive time-stepping for the strong numerical solution of stochastic differential equations." 
            #Numerical Algorithms 68.4 (2015): 791-812.
            z = torch.normal(mean=0.0, std=1.0, size=dW.shape, device = self.device, dtype=self.dtype)*dt**0.5
            dW_1 = 0.5*dW + 0.5*z
            dW_2 = 0.5*dW - 0.5*z
            ys_1 = self._step_forward_uncompiled(y_old, dW_1, dt/2, t)
            ys_2 = self._step_forward_uncompiled(ys_1, dW_2, dt/2, t)
            return (torch.abs(ys_2 - y_new).max()/self._tolerance)
        
    def set_stats(self, **kwargs):
        """This method is used to set the statistics thate need to be measured during the simulation.
            The data will be stored in a dictionary with keys mentioned below.
        
            The Following arguments are possible:
            > density = True/False, Calculate probability density. Keys: p(y,t), y, t
            > density_v = True/False, Calculate probability density of velocity. Keys: p(v,t), v, t
            > mean = True/False, Calculate mean. Keys: mean(t), t
            > mean_v = True/False, Calculate mean of velocity. Keys: mean_v(t), t
            > std = True/False, Calculate std. Keys: std(t), t
            > std_v = True/False, Calculate std of velocity. Keys: std_v(t), t
            > entropy = True/False, Calculate entropy. Keys: entropy(t), t
            > entropy_v = True/False, Calculate entropy of velocity. Keys: entropy_v(t), t
            > info_rate = True/False, Calculate information rate. Keys: info_rate(t*), t*
            > info_rate_v = True/False, Calculate information rate of velocity. Keys: info_rate(t#), t#
            > position = True/False, store the trajectory of the SDE. Keys: y(t), t
            
            > collection_frequency: int, frequency with which stats are collected. Default 1 (every 1 timestep).
            Note that if large number of parallel simulations are run position can quickly fill up the RAM.
            """
        
        all_stats = ['position', 'density', 'density_v', 'info_rate', 'mean', 'std', 'mean_v', 'std_v', 'entropy', 'entropy_v', 'info_rate_v', 'collection_frequency']
        
        ####Stat Variables##################
        self.collection_frequency = kwargs.get('collection_frequency', 1)
        
        for i in kwargs.keys():
            if i not in all_stats: raise Exception(f"{i} argument not recognized.")
        
        self._calc_position = kwargs.get('position', False)
        self._calc_density = kwargs.get('density', False)
        self._calc_density_v = kwargs.get('density_v', False)
        self._calc_info_rate = kwargs.get('info_rate', False)
        self._calc_mean = kwargs.get('mean', False)
        self._calc_std = kwargs.get('std', False)
        self._calc_mean_v = kwargs.get('mean_v', False)
        self._calc_std_v = kwargs.get('std_v', False)
        self._calc_entropy = kwargs.get('entropy', False)
        self._calc_entropy_v = kwargs.get('entropy_v', False)
        self._calc_info_rate_v = kwargs.get('info_rate_v', False)
        
        if self._calc_mean_v or self._calc_std_v or self._calc_entropy_v or self._calc_density_v or self._calc_info_rate_v:
            self._compute_velocity = True
            print("Warning: Stochastic processes doesn't have a well-defined instantaneous velocity. ", end="")
            print("Therefore a finite time limit is considered. Change value of 'velocity_avg_dt' attribute to change the finite time interval")
        else:
            self._compute_velocity = False
    
    def set_functions(self, f, g, df = None, dg = None, d2f = None, d2g = None):
        """This method is used to set the drift and diffusion functions in an SDE.
            > f: function, drift function
            > g: function, diffusion function
            > df: function, 1st derivative of drift function
            > dg: function, 1st derivative of diffusion function
            > d2f: function, 2nd derivative of drift function
            > d2g: function, 2nd derivative of diffusion function

            Currently only support functions containing basic operators and functions from python 'math' module.

            The derivative arguments are optional and will be consider as 0 if not provided.
            'euler_maruyama' method requires f and g.
            'milstein' method requires f, g and dg (default method).
            'taylor_15' method requires f, g, df, dg, d2f and d2g.

            If information rate of velocity is being measured, df is required. This will be fixed in the future.
        """
        self.fn = [f, g, df, dg, d2f, d2g]
        #In below code d variable used in lambda for compatibility with SDE_ND to pass time variable to step_forward function.
        if self.method == 'euler_maruyama':
            self._step_forward_uncompiled = lambda a,b,c,d: self._euler_maruyama(a,b,c, self.fn) 
        elif self.method == 'milstein' and self._functions_compiled == False:
            if dg == None: print("Warning: Assuming derivative of diffusion is 0 throughout the domain")
            self._step_forward_uncompiled = lambda a,b,c,d: self._milstein(a,b,c, self.fn)
        elif self.method == 'taylor_15':
            if self._functions_compiled == False:
                if dg == None: print("Warning: Assuming derivative of diffusion is 0 throughout the domain")
                if d2g == None: print("Warning: Assuming 2nd derivative of diffusion is 0 throughout the domain")
                if df == None: print("Warning: Assuming derivative of drift is 0 throughout the domain")
                if d2f == None: print("Warning: Assuming 2nd derivative of drift is 0 throughout the domain")
            self._step_forward_uncompiled = lambda a,b,c,d: self._taylor_15(a,b,c, self.fn)

        self.step_forward = torch.jit.trace(self._step_forward_uncompiled, (self.y_init, self.y_init, self.dt_init, self.dt_init), check_trace=False)
        
        if self.adaptive_stepping: self.step_error = torch.jit.trace(self._local_error, (self.y_init, self.y_init, self.y_init, self.dt_init, self.dt_init), check_trace=False)
        self.velocity = torch.jit.trace(self._velocity, (self.y_init, self.dt_init), check_trace = False)
        
        self._functions_compiled = True
    
    @torch.jit.script
    def hist2D(y_x, y_y, bins_x: int, bins_y: int, density: bool = False):
        '''Computes 2D histogram of joint probability.

           y_x: 1D tenosr, samples from x_axis.
           y_y: 1d tensor, corresponding samples from y_axis.
           bins_x: int, number of bins along x_axis.
           bins_y: int, number of bins aling y_axis.
           desnity: bool, True will return density rather than count.
        '''
        min_val_x = torch.min(y_x)
        max_val_x = torch.max(y_x)
        range_x = max_val_x-min_val_x
        x = torch.linspace(min_val_x, max_val_x, bins_x+1, dtype=y_x.dtype, device=min_val_x.device)
        x = (x[1:]+x[:-1])/2

        min_val_y = torch.min(y_y)
        max_val_y = torch.max(y_y)
        range_y = max_val_y-min_val_y
        y = torch.linspace(min_val_y, max_val_y, bins_y+1, dtype=y_y.dtype, device=min_val_y.device)
        y = (y[1:]+y[:-1])/2

        flat_y = torch.trunc((y_x-min_val_x)*(bins_x)/range_x)
        flat_y[flat_y==bins_x] = bins_x-1 #taking care of point on the right edge of the final bin
        flat_y += (y_y-min_val_y)/range_y

        hist = torch.histc(flat_y, min=0, max=bins_x, bins=int(bins_x*bins_y)).reshape((bins_x,bins_y))

        if density: hist /= (bins_x*bins_y*(x[1]-x[0])*(y[1]-y[0]))
        return x, y, hist.T
        
    @torch.jit.script
    def info_rate(y_new, y_old, dt, bins: int = 0, gauss_estimate : bool = True):
        """Compute information rate.
           > 
           
           returns: (Sqrt approx, Symmetrized KL estimate, Gauss Approx (if gauss_estimate=True))
        """
        num_par = y_new.shape[-1]
        if bins==0: bins = int(1.4*2.59*num_par**0.33333)
    
        min_val_new = torch.min(y_new, dim=-1)[0]
        min_val_old = torch.min(y_old, dim=-1)[0]
        max_val_new = torch.max(y_new, dim=-1)[0]
        max_val_old = torch.max(y_old, dim=-1)[0]
        min_val = torch.min(min_val_old,min_val_new)
        max_val = torch.max(max_val_old,max_val_new)

        prob_old = histc(y_old, bins=bins, max=max_val, min=min_val )/num_par
        prob_new = histc(y_new, bins=bins, max=max_val, min=min_val)/num_par
        ir_approx = 4*((prob_old**0.5-prob_new**0.5)**2).sum(dim=-1)

        index = (prob_new-prob_old)*(prob_new!=0)*(prob_old!=0) #removing zeros with nansum in next line
        kl_symm = (index*(torch.log(prob_new)-torch.log(prob_old))).nansum(dim=-1)
    
        if gauss_estimate:
            var_new = torch.var(y_new, unbiased = True, dim=-1)
            var_old = torch.var(y_old, unbiased = True, dim=-1)
            mean_new = torch.mean(y_new, dim=-1)
            mean_old = torch.mean(y_old, dim=-1)
            info_rate_normal = (var_new+(mean_new-mean_old)**2)/(var_old*2)+(var_old+(mean_new-mean_old)**2)/(var_new*2)-1
            return torch.stack((ir_approx, kl_symm, info_rate_normal), dim=-1)**0.5/dt
        else:
            return torch.stack((ir_approx, kl_symm), dim=-1)**0.5/dt

    @torch.jit.script
    def histogram(item, bins: int = 0, density: bool = False):
        min_val = torch.min(item, dim=-1)[0]
        max_val = torch.max(item, dim=-1)[0]
        
        if bins==0: bins = int(2.59*item.shape[-1]**0.33333)
        prob = histc(item, bins=bins, max=max_val, min =min_val)
        
        dx = (max_val-min_val)/bins
        if len(item.shape)>1:
            bin_edge  = min_val[:,None] + torch.arange(bins+1, device=min_val.device, dtype=min_val.dtype)[None,:]*dx[:,None]
            bin_centre = (bin_edge[:,1:]+bin_edge[:,:-1])/2
            if density: prob = torch.nan_to_num(prob/(item.shape[-1]*dx[:, None]), nan=0.0, posinf=float('inf'))
            return prob, bin_centre
        else:
            bin_edge  = min_val + torch.arange(bins+1, device=min_val.device, dtype=min_val.dtype)*dx
            bin_centre = (bin_edge[1:]+bin_edge[:-1])/2
            if density: prob = torch.nan_to_num(prob/(item.shape[-1]*dx), nan=0.0, posinf=float('inf'))
            return prob, bin_centre
    
    @torch.jit.script
    def entropy(item, bins: int = 0):
        min_val = torch.min(item, dim=-1)[0]
        max_val = torch.max(item, dim=-1)[0]

        if bins==0:
                bins = int(2.59*item.shape[-1]**0.33333)

        dx = (max_val-min_val)/bins
        prob = histc(item, bins=bins, max=max_val, min =min_val)/item.shape[-1]
        index = prob!=0
        return -torch.nansum((prob*torch.log(prob)), dim=-1) + torch.log(dx)

    def _compute_info_rate_v(self, v_new, y, dt):
        if self.dt_elapsed_v == 0:
            v = self.velocity(y, self.t)
            self.t_old_v = self.t
            self.dt_old_v = dt
            self.v_old = v
            self.v_mean = self.data['mean_v(t)'][-2] if self._calc_mean_v else torch.mean(v)
            self.v_std = self.data['std_v(t)'][-2] if self._calc_std_v else torch.std(v, unbiased = True)
            self.dt_interp_v = self._mean_shift_threshold*self.v_std/torch.abs((self.fn[0](y)*self.fn[2](y)).mean())

        self.dt_elapsed_v += dt   

        new_std = self.data['std_v(t)'][-1] if self._calc_std_v else torch.std(v_new, unbiased = True)
        if self.dt_interp_v<self.dt_old_v:
            xi = torch.normal(mean=0.0, std=self.dt_interp_v**0.5, size=(self.num_par,), device = self.device, dtype=self.dtype)
            y_interp = self.step_forward(y, xi, self.dt_interp_v, self.t)
            v_interp = self.velocity(y_interp, self.t)
            
            self.data['info_rate_v(t#)'].append(self.info_rate(v_interp, self.v_old, self.dt_interp_v, gauss_estimate = self.gauss_estimate))
            self.data['t#'].append(self.t_old_v + 0.5*self.dt_interp_v)
            
            if self.debug:
                print(f"V::IR:Interp: time={self.data['t#'][-1]:.3e}, dt_interp={self.dt_interp_v:.3e}, mean_diff={torch.abs(torch.mean(v_interp)-self.v_mean):.3e}, std={self.v_std:.3e}")
                            
            self.dt_elapsed_thresh_v = self.dt_elapsed_v*1.2
            self.dt_elapsed_v *= 0
        
        elif torch.abs(torch.mean(v_new)-self.v_mean) > self._mean_shift_threshold*self.v_std:
            #there will be a small loss of accuracy if the overlap is small
            self.data['info_rate_v(t#)'].append(self.info_rate(v_new, self.v_old, self.dt_elapsed_v, gauss_estimate = self.gauss_estimate))
            self.data['t#'].append(self.t_old_v + 0.5*self.dt_elapsed_v)
            
            if self.debug:
                print(f"V::IR:Mean: time={self.data['t#'][-1]:.3e}, dt_elapsed={self.dt_elapsed_v:.3e}, mean_diff={torch.abs(torch.mean(v_new)-self.v_mean):.3e}, std={self.v_std:.3e}, mean={self.v_mean:.3e}")
                
            self.dt_elapsed_thresh_v = self.dt_elapsed_v*1.2
            self.dt_elapsed_v *= 0

        elif self.v_std/new_std<0.9 or new_std/self.v_std<0.9:    
            self.data['info_rate_v(t#)'].append(self.info_rate(v_new, self.v_old, self.dt_elapsed_v, gauss_estimate = self.gauss_estimate))
            self.data['t#'].append(self.t_old_v + 0.5*self.dt_elapsed_v)
            
            if self.debug:
                print(f"V::IR:Std_: time={self.data['t#'][-1]:.3e}, dt_elapsed={self.dt_elapsed:.3e}, std_old:std_new={self.v_std/new_std:.3e}, std={self.v_std:.3e}, mean={self.v_mean:.3e}")
            
            self.dt_elapsed_thresh_v = self.dt_elapsed_v*1.2
            self.dt_elapsed_v *= 0

        elif self.dt_elapsed_thresh_v < self.dt_elapsed_v:
            self.data['info_rate_v(t#)'].append(self.info_rate(v_new, self.v_old, self.dt_elapsed_v, gauss_estimate = self.gauss_estimate))
            self.data['t#'].append(self.t_old_v + 0.5*self.dt_elapsed_v)
            
            if self.debug:
                print(f"V::IR:Iter: time={self.data['t#'][-1]:.3e}, dt_elapsed={self.dt_elapsed_v:.3e}, dt_thres={self.dt_elapsed_thresh_v:.3e}, std={self.v_std:.3e}, mean={self.v_mean:.3e}")
                
            self.dt_elapsed_thresh_v = torch.clamp(self.dt_elapsed_thresh_v*1.2, min = self.tolerance, max = 20.0)
            self.dt_elapsed_v *= 0
            
    def _compute_info_rate(self, y_new, y, dt):
        if self.dt_elapsed==0:
            self.t_old = self.t
            self.y_old = y
            self.y_next = y_new
            self.dt_old = dt
            self.y_mean = self.data['mean(t)'][-2] if self._calc_mean else torch.mean(y)
            self.y_std = self.data['std(t)'][-2] if self._calc_std else torch.std(y, unbiased = True)
            self.dt_interp = self._mean_shift_threshold*self.y_std/torch.abs(self.fn[0](y).mean())

        self.dt_elapsed+=dt   

        new_std = self.data['std(t)'][-1] if self._calc_std else torch.std(y_new, unbiased = True)
        
        if self.dt_interp<self.dt_old: #questionable step
            dW_interp = torch.normal(mean=0.0, std=torch.sqrt(self.dt_interp), size=(self.num_par,), device = self.device, dtype=self.dtype)
            y_interp = self.step_forward(y, dW_interp, self.dt_interp, self.t)
            self.data['info_rate(t*)'].append(self.info_rate(y_interp, y, self.dt_interp, gauss_estimate = self.gauss_estimate))
            self.data['t*'].append(self.t +0.5*self.dt_interp)
            
            if self.debug:
                print(f"IR:Interp: time={self.t:.3e}, dt_interp={self.dt_interp:.3e}, mean_diff={torch.abs(torch.mean(y_interp)-self.y_mean):.3e}, std={self.y_std:.3e}")
            
            self.dt_elapsed_thresh = self.dt_elapsed*1.2
            self.dt_elapsed *= 0

        elif torch.abs(torch.mean(y_new)-self.y_mean) > self._mean_shift_threshold*self.y_std:
            #there will be a small loss of accuracy if the overlap is small
            self.data['info_rate(t*)'].append(self.info_rate(y_new, self.y_old, self.dt_elapsed, gauss_estimate = self.gauss_estimate))
            self.data['t*'].append(self.t_old + 0.5*self.dt_elapsed)
            
            if self.debug:
                print(f"IR:Mean: time={self.t:.3e}, dt_elapsed={self.dt_elapsed:.3e}, mean_diff={torch.abs(torch.mean(y_new)-self.y_mean):.3e}, std={self.y_std:.3e}, mean={self.y_mean:.3e}")
                
            self.dt_elapsed_thresh = self.dt_elapsed*1.2
            self.dt_elapsed *= 0

        elif self.y_std/new_std<0.9 or new_std/self.y_std<0.9:
            if self.debug:
                print(f"IR:Std_: time={self.t:.3e}, dt_elapsed={self.dt_elapsed:.3e}, std_old:std_new={self.y_std/new_std:.3e}, std={self.y_std:.3e}, mean={self.y_mean:.3e}")
                
            self.data['info_rate(t*)'].append(self.info_rate(y_new, self.y_old, self.dt_elapsed, gauss_estimate = self.gauss_estimate))
            self.data['t*'].append(self.t_old + 0.5*self.dt_elapsed)
            
            self.dt_elapsed_thresh = self.dt_elapsed*1.2
            self.dt_elapsed *= 0

        elif self.dt_elapsed_thresh < self.dt_elapsed:
            if self.debug:
                print(f"IR:Iter: time={self.t:.3e}, dt_elapsed={self.dt_elapsed:.3e}, dt_thres={self.dt_elapsed_thresh:.3e}, std={self.y_std:.3e}, mean={self.y_mean:.3e}")
                
            self.data['info_rate(t*)'].append(self.info_rate(y_new, self.y_old, self.dt_elapsed, gauss_estimate = self.gauss_estimate))
            self.data['t*'].append(self.t_old + 0.5*self.dt_elapsed)
            
            self.dt_elapsed_thresh = torch.clamp(self.dt_elapsed_thresh*1.2, min = self.tolerance, max = 20.0)
            self.dt_elapsed *= 0
    
    def _collect_stats(self, y_new, y, dt, init=False):
        if init == False and self.i%self.collection_frequency != 0:
            return
    
        if self._compute_velocity:
            #xi = torch.normal(mean=0.0, std=1e-4, size=(self.num_par,), device = self.device, dtype=self.dtype)
            v_new = self.velocity(y_new, self.t)
        
        if self._calc_position: self.data['y(t)'].append(y_new.to('cpu'))
        if self._calc_mean: self.data['mean(t)'].append(torch.mean(y_new, dim=-1))
        if self._calc_std: self.data['std(t)'].append(torch.std(y_new, unbiased = True, dim=-1))
        if self._calc_mean_v: self.data['mean_v(t)'].append(torch.mean(v_new, dim=-1))
        if self._calc_std_v: self.data['std_v(t)'].append(torch.std(v_new, unbiased = True, dim=-1)) 
        ##std and mean should be always above other stats
        if self._calc_entropy: self.data['entropy(t)'].append(self.entropy(y_new))
        if self._calc_entropy_v: self.data['entropy_v(t)'].append(self.entropy(v_new))
            
        if self._calc_density: 
            prob_density, bin_centre = self.histogram(y_new, density=True)
            self.data['p(y,t)'].append(prob_density.to('cpu'))
            self.data['y'].append(bin_centre.to('cpu'))
        if self._calc_density_v: 
            prob_density, bin_centre = self.histogram(v_new, density=True)
            self.data['p(v,t)'].append(prob_density.to('cpu')) #send to cpu. Can get too large for GPU
            self.data['v'].append(bin_centre.to('cpu'))
            
        if init: 
            self.data['t'].append(self.t)
            if self._calc_info_rate:
                self.dt_elapsed = torch.tensor(0.0, dtype = self.dtype, device = self.device)
                self.dt_elapsed_thresh = torch.tensor(1e4, dtype = self.dtype, device = self.device)
            if self._calc_info_rate_v:
                self.dt_elapsed_v = torch.tensor(0.0, dtype = self.dtype, device = self.device)
                self.dt_elapsed_thresh_v = torch.tensor(1e4, dtype = self.dtype, device = self.device)
            return
        
        if self._calc_info_rate: self._compute_info_rate(y_new, y, dt)  
        if self._calc_info_rate_v: 
            self._compute_info_rate_v(v_new, y, dt)
                
        self.data['t'].append(self.t + dt)
            
    def _initialize_y(self, init_mean, init_std):
        return torch.normal(mean = init_mean, std = init_std, size = self.y_init.shape, device = self.device,
                                       dtype = self.dtype)

    def simulate(self, t_init : float, t_end : float, init_mean : float, init_std : float):
        """
        > t_init : float, inital time 
        > t_end : float, final time
        > init_mean : float, mean of initial distribution
        > init_std : float, standard deviation of initial distribution
        """

        self.data = defaultdict(list)
        if self.debug == True: self.debug_output = []
        ######### physical parameters #############
        self.t_init = torch.tensor(t_init, dtype = self.dtype, device = self.device) #initial time
        self.t_end  = torch.tensor(t_end, dtype = self.dtype, device = self.device)  #final time
        
        print("Starting Simulation...")
        self.accept, self.reject, self.i = 0, 0, -1
        self.t = self.t_init
        dt = self.dt_init
        self.y_init = y = self._initialize_y(init_mean, init_std)
        
        self._collect_stats(y, y, dt, init=True)
        
        t1 = time.perf_counter()
        while(self.t < self.t_end):
            self.i+=1     
            ################# Milstein method #################
            dW = torch.normal(mean=0.0, std=torch.sqrt(dt), size=self.y_init.shape, device = self.device, dtype=self.dtype)
            y_new = self.step_forward(y, dW, dt, self.t)

            if self.adaptive_stepping:
                err = self.step_error(y_new, y, dW, dt, self.t)
                if err>=1:
                    self.reject+=1
                    dt = dt*torch.clamp(torch.pow(self._fac/err, self._err_scale_exp), self._facmin, self._facmax) #1.5 for milstein methd
                    continue ###Discontinuing current iteration

            ########Compute Statistics##############################
            self._collect_stats(y_new, y, dt)

            y = y_new
            self.accept+=1
            self.t = self.t+dt
            
            if self.adaptive_stepping: dt = dt*torch.clamp(torch.pow(self._fac/err, self._err_scale_exp), self._facmin, self._facmax)

            #updating progress bar
            if self.debug == False and (self.i%20 == 0 or self.t >= self.t_end):
                eta = (self.t_end - self.t)*(time.perf_counter() - t1)/(self.t - self.t_init)
                print("<"+"="*int(self.t*50/(self.t_end - self.t_init)) + "_"*int(50*(1 - self.t/(self.t_end-  self.t_init)))+">"\
                     + " dt: " + f"{dt:.2e}" + f" acc: {self.accept} rej: {self.reject}"\
                     + f" ETA: {eta:.0f}s" + ' '*10, end='\r')
        
        t2 = time.perf_counter()
        print(f"\nSimulation Finished: {t2-t1:.2f}s")
        
        self.t1, self.t2 = t1, t2
        for item in self.data.keys():
            self.data[item] = torch.stack(self.data[item]).to('cpu').numpy()
    
    def _save_functions(self):
        self.data['functions'] = {}
        for i, fn_str in enumerate(self.fn_def_strings):
            if self.fn[i] == None:
                self.data['functions'][fn_str] = 'None'
            else:
                self.data['functions'][fn_str] = inspect.getsource(self.fn[i])
        
    def save_data(self, output_directory, filename):
        """The data is saved as a dictionary. Use pickle to load the saved data.
            info_rate is saved as an multi dimensional array with each column representing 2 different estimates.
            2nd column calculated using symmetrized KL diveregence. 1st column using an approximation of this expression.
            If the attribute guass_estimate = True, 3rd column represents an approximation assuming gaussian distribution.
        """
        parameters = {'t_init': self.t_init.item(), 't_end': self.t_end.item(), "dt_init": self.dt_init.item(), "time_taken": self.t2-self.t1,
                      'num_par': self.num_par, "accepted": self.accept, "rejected": self.reject,
                      'tolerance': self._tolerance.item(), "fac": self._fac.item(), "facmin": self._facmin.item(), "facmax": self._facmax.item()
                      }
        self.data['parameters'] = parameters
        self._save_functions()
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(output_directory,filename)
        with open(file_path, 'wb') as f:
            pickle.dump(dict(self.data), f) #dict() convert defaultdict to dict
            print(f"Output: {file_path}, Size: {os.path.getsize(file_path)/1024**2 :3f} MB")