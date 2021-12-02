import numpy as np
import time
import torch
import torch.multiprocessing as mp
import json
from dependencies import *

# Given the following SDE:
# dY_t = f(Y_t)dt + g(Y_t).dW_t
# Where W_t is the Weiner process
#
# Using Milstein method we have:
# Y_(t+dt) = Y_t + f(Y_t).dt + g(Y_t).dW_t + 1/2*g(Y_t)*g'(Y_t)((dW_t)^2 - dt)


def milstein(Y_t, dW_t, dt: float):
    sigma = 1.0
    def f(y): return -4*(y)**3 + 4*(y) #Drift term
    def g(y): return sigma*y #Diffusion term
    def dg(y): return sigma  #Derivative of Diffusion term

    ode_term = Y_t + f(Y_t)*dt
    stochastic_term = g(Y_t)*dW_t + 0.5*g(Y_t)*dg(Y_t)*(dW_t**2 - dt)
    return ode_term + stochastic_term





if __name__ == '__main__':
    output_directory = 'data/'
    task = 'find_density' #keep unchanged.
    cpu_buffer_size = 20 #memory buffer for VRAM->RAM (GPU->CPU) data transfer
    mem_height = 10 #size of datastructure in GPU. Minimum: 2
    #array size = mem_height x num_par
    num_par = 1_000_000 #number of parallel simulations

    #Error calculation and Adaptive stepping parameters.
    tolerance = 1e-3
    fac = 0.8
    facmin = 0.5
    facmax = 1.2

    # physical parameters
    t_init = 0  #initial time
    t_end  = 10  #final time
    dt_init = 0.001 #initial dt
    #grid_size  = 10000 # Number of grid points
    ## Initial Conditions
    y_init = torch.normal(mean=2.0, std=np.sqrt(2.0), size=(num_par,), device = torch.device('cuda'), dtype=torch.float32)
     
    assert(mem_height>=2) 

    torch.cuda.empty_cache()
    q1 = mp.Queue(maxsize=cpu_buffer_size)

    
    print("Creating Large Arrays....")
    ys_cpu_buffer = torch.zeros(cpu_buffer_size, mem_height, num_par, dtype = torch.float32, device = torch.device('cpu'), pin_memory=True)
    ys_cpu_buffer.share_memory_()
    ys = torch.zeros(mem_height, num_par, dtype = torch.float32, device = torch.device('cuda'))

    print(f"GPU Memory: {tensor_memory(ys):.2f} MB")
    print(f"CPU Memory: {tensor_memory(ys_cpu_buffer[0])*cpu_buffer_size:.2f} MB")

    if task == 'write_data':
        threads = min(10, mp.cpu_count()) #number of writing threads RAM->Harddrive
        proc = [mp.Process(target = writer, args = (q1, ys_cpu_buffer,)) for _ in range(threads)]
        for item in proc: item.start()
    elif task == 'find_density':
        p_dict = mp.Manager().dict()
        proc = [mp.Process(target = calc_density, args = (q1, ys_cpu_buffer, p_dict)), ]
        for item in proc: item.start()
        while 'start' not in p_dict: time.sleep(0.1)
    else: 
        proc = [] 
    
    # Loop
    print("Starting Simulation...")
    buffer_index, num_parts, i, t = 0, 0, 1, 0
    accept, reject = 0, 0 
    t1 = time.perf_counter()
    ys[0][:] = y_init
    t_list = [t_init, ]
    dt = dt_init
    while(t < t_end):
            
        #updating progress bar
        if i%20 == 0:
               print("<"+"="*int(t*50/(t_end-t_init))+"_"*int(50*(1-t/(t_end-t_init)))+">"\
                     + " dt: " + f"{dt:.5f}" + f" acc: {accept} rej: {reject}", end='\r')
        
        ################# Milstein method #################
        dW = torch.normal(mean=0.0, std=np.sqrt(dt), size=(num_par,), device = torch.device('cuda'), dtype=torch.float32)
        ys_0 = ys[i%mem_height] = milstein(ys[(i-1)%mem_height], dW, dt)

            
        ################# Calculating Error ################
        #Ilie, Silvana, Kenneth R. Jackson, and Wayne H. Enright. 
        #"Adaptive time-stepping for the strong numerical solution of stochastic differential equations." 
        #Numerical Algorithms 68.4 (2015): 791-812.
        z = torch.normal(mean=0.0, std=np.sqrt(dt), size=(num_par,), device = torch.device('cuda'), dtype=torch.float32)
        dW_1 = 0.5*dW + 0.5*z
        dW_2 = 0.5*dW - 0.5*z
        ys_1 = milstein(ys[(i-1)%mem_height], dW_1, dt/2)
        ys_2 = milstein(ys_1, dW_2, dt/2)
        err = float(torch.abs(ys_2 - ys_0).max().to('cpu')/tolerance)

        if err>=1:
            reject+=1
            dt/=2
            continue

        accept+=1
        t+=dt
        i+=1
        t_list.append(t)
        dt = dt*np.clip(np.power(fac/err, 1/(1.5)), facmin, facmax)
        ################# Data Transfer: GPU -> CPU #################
        if (i)%mem_height==0 or t>=t_end:
            ys_cpu_buffer[buffer_index].copy_(ys, non_blocking=True)
            if proc != []: q1.put((buffer_index, t_list))
            buffer_index = (buffer_index+1)%cpu_buffer_size
            num_parts+=1
            t_list = []
                
        
        

    t2 = time.perf_counter()
    print(f"\nSimulation Time: {t2-t1:.2f}s")
    
    ################# Cleaning Up #################
    del ys, dW, ys_cpu_buffer
    q1.put(('e','e')) #termination sequence for processes
    for item in proc:
        item.join()
        item.close()
    while not q1.empty(): q1.get(block = False)
    q1.close()

    ################# Saving Parameters #################
    print("Saving Parameters...")
    parameters = {'t_init': t_init, 't_end': t_end, "dt_init": dt_init,
                  'num_par': num_par, "accepted": accept, "rejected": reject,
                  'cpu_buffer_size': cpu_buffer_size,  'mem_height': mem_height,
                  'tolerance': tolerance
                  }
    with open(output_directory + "parameters.txt", 'w') as file:
        json.dump(parameters, file)
    if task == 'find_density':
        np.save(output_directory + 'p_x.npy', p_dict['p_x'])
        np.save(output_directory + 'x.npy', p_dict['x'])
        np.save(output_directory + 't.npy', p_dict['t'])

    t2 = time.perf_counter()
    print(f"Finished: {t2-t1:.2f}s")