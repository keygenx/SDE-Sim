import numpy as np
import time
import torch
import torch.multiprocessing as mp
import json
import inspect
from dependencies import *



############################################ Simulation Parameters ######################################################
#########################################################################################################################
output_directory = 'data_test/0.6/'
task = 'find_density' #keep unchanged.
num_par = 10_000_000 #number of parallel simulations
gpu_buffer_size = 3 #size of datastructure in GPU. Minimum: 2. array size = gpu_buffer_size x num_par

####Error calculation and Adaptive stepping parameters.
adaptive_stepping = True #use adaptive stepping
dt_init = 0.0001 #initial step size
tolerance = 1e-3 #Maximum allowed local error.
fac = 0.9
facmin = 0.2
facmax = 1.3

######### physical parameters #############
t_init = 0.0  #initial time
t_end  = 10.0  #final time

D = 1.0
gamma = 0.2
def f_x(y): return -(y)**3 - (y) #Drift term
def g_x(y): 
    D_eta = 0.6
    return torch.sqrt(2*D_eta) #Diffusion term
def dg_x(y): return 0.0  #Derivative of Diffusion term

def f_f(y): 
    gamma=0.2
    return -gamma*y #Drift term
def g_f(y):
    D=1.0
    return torch.sqrt(2*D) #Diffusion term
def dg_f(y): return 0.0  #Derivative of Diffusion term

## Initial distribution
x0 = 1.2
x_init = torch.normal(mean=x0, std=np.sqrt(0.5*10**(-3)), size=(num_par,), device = torch.device('cuda'), dtype=torch.float32)
f_init = torch.normal(mean=f_x(x0), std=np.sqrt(D/gamma), size=(num_par,), device = torch.device('cuda'), dtype=torch.float32)
#########################################################################################################################
#########################################################################################################################


@torch.jit.script
def milstein_x(x_old, d_eta, f, dt: float):
    ode_term = x_old + (f_x(x_old)+f)*dt 
    stochastic_term = g_x(x_old)*d_eta#+ 0.5*g_x(x_old)*dg_x(f_old)*(d_eta**2 - dt)
    return ode_term + stochastic_term

@torch.jit.script
def milstein_f(f_old, d_xi, dt: float):
    ode_term = f_old + f_f(f_old)*dt
    stochastic_term = g_f(f_old)*d_xi #+ 0.5*g_f(f_old)*dg_f(f_old)*(d_eta**2 - dt)
    return ode_term + stochastic_term

@torch.jit.script
def local_error(x_new, x_old, f_new, f_old, d_eta, d_xi, dt: float, tolerance: float):
    z = torch.normal(mean=0.0, std=torch.sqrt(dt), size=d_eta.shape, device = torch.device('cuda'), dtype=torch.float32)
    d_eta_1 = 0.5*d_eta + 0.5*z
    d_eta_2 = 0.5*d_eta - 0.5*z
    z = torch.normal(mean=0.0, std=torch.sqrt(dt), size=d_xi.shape, device = torch.device('cuda'), dtype=torch.float32)
    d_xi_1 = 0.5*d_xi + 0.5*z
    d_xi_2 = 0.5*d_xi - 0.5*z 

    f_1 = milstein_f(f_old, d_xi_1, dt/2)
    f_2 = milstein_f(f_1, d_xi_2, dt/2)
    x_1 = milstein_x(x_old, d_eta_1, f_old, dt/2)
    x_2 = milstein_x(x_1, d_eta_2, f_1, dt/2)

    return float((torch.sqrt(torch.max((x_2-x_new)**2 + (f_2-f_new)**2)/2)/tolerance).cpu())

if __name__ == '__main__':
   
    assert gpu_buffer_size>=2, "Error: gpu_buffer_size < 2"
    assert t_end>t_init, "Error: t_end < t_init"
    assert facmin<1.0 and facmax>1.0 and fac<=1.0, "Adaptive stepping parameter error"

    torch.manual_seed(0)
    torch.cuda.empty_cache()
    
    print("Creating Large Arrays....")
    xs = torch.zeros(gpu_buffer_size, num_par, dtype = torch.float32, device = torch.device('cuda'))
    fs = torch.zeros(gpu_buffer_size, num_par, dtype = torch.float32, device = torch.device('cuda'))
    print(f"GPU Memory: {tensor_memory(xs)*2:.2f} MB")
   
    
    ################### Main Loop ###################################
    print("Starting Simulation...")
    accept, reject, i = 0, 0, 0

    prob_density, bin_centre = cuda_histogram(x_init, bins=400)
    prob_densities, bin_centres, t_list = [prob_density], [bin_centre], [t_init]
    t, dt = t_init, dt_init

    xs[0][:] = x_init
    fs[0][:] = f_init

    t1 = time.perf_counter()
    while(t < t_end):
        i+=1  
        
        ################# Milstein method #################
        d_xi = torch.normal(mean=0.0, std=np.sqrt(dt), size=(num_par,), device = torch.device('cuda'), dtype=torch.float32)
        d_eta = torch.normal(mean=0.0, std=np.sqrt(dt), size=(num_par,), device = torch.device('cuda'), dtype=torch.float32)
        f = fs[(i-1)%gpu_buffer_size]
        x = xs[(i-1)%gpu_buffer_size]
        f_new = fs[i%gpu_buffer_size] = milstein_f(f, d_xi, dt)
        x_new = xs[i%gpu_buffer_size] = milstein_x(x, d_eta, f, dt)

        if adaptive_stepping:
            err = local_error(x_new, x, f_new, f, d_eta, d_xi, dt, tolerance)
            if err>=1:
                reject+=1
                dt = dt*np.clip(np.power(fac/err, 1/(1.5)), facmin, facmax)
                continue ###Discontinuing current iteration


        accept+=1
        t+=dt
        if adaptive_stepping: dt = dt*np.clip(np.power(fac/err, 1/(1.5)), facmin, facmax)
        
        ################# Probability Density Calculation ######
        prob_density, bin_centre = cuda_histogram(x_new, bins=400)
        prob_densities.append(prob_density)
        bin_centres.append(bin_centre)
        t_list.append(t)

        #updating progress bar
        if i%20 == 0 or t>=t_end:
            eta = (t_end-t)*(time.perf_counter()-t1)/(t-t_init)
            print("<"+"="*int(t*50/(t_end-t_init))+"_"*int(50*(1-t/(t_end-t_init)))+">"\
                     + " dt: " + f"{dt:.2e}" + f" acc: {accept} rej: {reject}"\
                     + f" ETA: {eta:.0f}s", end='\r')
                
    t2 = time.perf_counter()
    print(f"\nSimulation Finished: {t2-t1:.2f}s")
    
    ################# Cleaning Up #################
    del xs, fs, d_eta, d_xi

    ################# Saving Data #####################
    np.save(output_directory + 'p_x.npy', torch.stack(prob_densities).cpu().numpy())
    np.save(output_directory + 'x.npy', torch.stack(bin_centres).cpu().numpy())
    np.save(output_directory + 't.npy', np.array(t_list))

    print("Saving Parameters...")
    parameters = {'t_init': t_init, 't_end': t_end, "dt_init": dt_init, "time_taken": t2-t1,
                  'num_par': num_par, "accepted": accept, "rejected": reject,
                  'gpu_buffer_size': gpu_buffer_size,
                  'tolerance': tolerance, "fac": fac, "facmin": facmin, "facmax": facmax,
                  'f_f()': inspect.getsource(f_f), 'g_f()': inspect.getsource(g_f), 'dg_f()': inspect.getsource(dg_f),
                  'f_x()': inspect.getsource(f_x), 'g_x()': inspect.getsource(g_x), 'dg_x()': inspect.getsource(dg_x),
                  }
    with open(output_directory + "parameters.txt", 'w') as file:
        json.dump(parameters, file)

    t2 = time.perf_counter()
    print(f"Total Time: {t2-t1:.2f}s")