import numpy as np
import time
import torch
import torch.multiprocessing as mp
import json
import inspect
from dependencies import *

# Given the following SDE:
# dY_t = f(Y_t)dt + g(Y_t).dW_t
# Where W_t is the Weiner process
#
# Using Milstein method we have:
# Y_(t+dt) = Y_t + f(Y_t).dt + g(Y_t).dW_t + 1/2*g(Y_t)*g'(Y_t)((dW_t)^2 - dt)

############################################ Simulation Parameters ######################################################
#########################################################################################################################
output_directory = 'data_test/'
task = 'find_density' #keep unchanged.
num_par = 100_000_000 #number of parallel simulations
gpu_buffer_size = 3 #size of datastructure in GPU. Minimum: 2. array size = mem_height x num_par

####Error calculation and Adaptive stepping parameters.
adaptive_stepping = True #use adaptive stepping
dt_init = 0.001 #initial step size
tolerance = 1e-4 #Maximum allowed local error.
fac = 0.9
facmin = 0.2
facmax = 1.3

######### physical parameters #############
t_init = 0.0  #initial time
t_end  = 4.0  #final time

def f(y): return -2*(y) #Drift term
def g(y): return 2**0.5 #Diffusion term
def dg(y): return 0.0  #Derivative of Diffusion term

## Initial distribution
y_init = torch.normal(mean=-(1.2)**3 - (1.2), std=np.sqrt(1/0.2), size=(num_par,), device = torch.device('cuda'), dtype=torch.float32)
#########################################################################################################################
#########################################################################################################################


@torch.jit.script
def milstein(Y_t, dW_t, dt: float):
    ode_term = Y_t + f(Y_t)*dt
    stochastic_term = g(Y_t)*dW_t + 0.5*g(Y_t)*dg(Y_t)*(dW_t**2 - dt)
    return ode_term + stochastic_term

@torch.jit.script
def local_error(y_new, y_old, dW, dt: float, tolerance: float):
    #Ilie, Silvana, Kenneth R. Jackson, and Wayne H. Enright. 
    #"Adaptive time-stepping for the strong numerical solution of stochastic differential equations." 
    #Numerical Algorithms 68.4 (2015): 791-812.
    z = torch.normal(mean=0.0, std=torch.sqrt(dt), size=dW.shape, device = torch.device('cuda'), dtype=torch.float32)
    dW_1 = 0.5*dW + 0.5*z
    dW_2 = 0.5*dW - 0.5*z
    ys_1 = milstein(y_old, dW_1, dt/2)
    ys_2 = milstein(ys_1, dW_2, dt/2)
    return float(torch.abs(ys_2 - y_new).max().cpu()/tolerance)

if __name__ == '__main__':
   
    assert gpu_buffer_size>=2, "Error: gpu_buffer_size < 2"
    assert t_end>t_init, "Error: t_end < t_init"
    assert facmin<1.0 and facmax>1.0 and fac<=1.0, "Adaptive stepping parameter error"

    torch.manual_seed(0)
    torch.cuda.empty_cache()
    
    print("Creating Large Arrays....")
    ys = torch.zeros(gpu_buffer_size, num_par, dtype = torch.float32, device = torch.device('cuda'))
    print(f"GPU Memory: {tensor_memory(ys):.2f} MB")


    ################### Main Loop ###################################
    print("Starting Simulation...")
    accept, reject, i = 0, 0, 0
    
    prob_density, bin_centre = cuda_histogram(y_init, bins=400)
    prob_densities, bin_centres, t_list = [prob_density], [bin_centre], [t_init]
    t, dt, ys[0][:] = t_init, dt_init, y_init

    t1 = time.perf_counter()
    while(t < t_end):
        i+=1      
        ################# Milstein method #################
        dW = torch.normal(mean=0.0, std=np.sqrt(dt), size=(num_par,), device = torch.device('cuda'), dtype=torch.float32)
        y_old = ys[(i-1)%gpu_buffer_size]
        y_new = ys[i%gpu_buffer_size] = milstein(y_old, dW, dt)

        if adaptive_stepping:
            err = local_error(y_new, y_old, dW, dt, tolerance)
            if err>=1:
                reject+=1
                dt = dt*np.clip(np.power(fac/err, 1/(1.5)), facmin, facmax)
                continue ###Discontinuing current iteration

        accept+=1
        t+=dt
        if adaptive_stepping: dt = dt*np.clip(np.power(fac/err, 1/(1.5)), facmin, facmax)
        
        ################# Probability Density Calculation ######
        prob_density, bin_centre = cuda_histogram(y_new, bins=400)
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
    del ys, dW

    ################# Saving Data #####################
    np.save(output_directory + 'p_x.npy', torch.stack(prob_densities).cpu().numpy())
    np.save(output_directory + 'x.npy', torch.stack(bin_centres).cpu().numpy())
    np.save(output_directory + 't.npy', np.array(t_list))

    print("Saving Parameters...")
    parameters = {'t_init': t_init, 't_end': t_end, "dt_init": dt_init, "time_taken": t2-t1,
                  'num_par': num_par, "accepted": accept, "rejected": reject,
                  'gpu_buffer_size': gpu_buffer_size, 
                  'tolerance': tolerance, "fac": fac, "facmin": facmin, "facmax": facmax,
                  'f(Y)': inspect.getsource(f), 'g(y)': inspect.getsource(g), 'dg(Y)': inspect.getsource(dg)
                  }
    with open(output_directory + "parameters.txt", 'w') as file:
        json.dump(parameters, file)

    t2 = time.perf_counter()
    print(f"Total Time: {t2-t1:.2f}s")