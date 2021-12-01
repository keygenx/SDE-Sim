import numpy as np
import time
import torch
import torch.multiprocessing as mp
import json
from dependencies import *

#Force
def dV(x):
    return 4*(x)**3-4*(x)

if __name__ == '__main__':
    output_directory = 'data/'
    task = 'find_density' #keep unchanged.
    threads = 10 #number of writing threads RAM->Harddrive
    cpu_buffer_size = 20 #memory buffer for VRAM->RAM (GPU->CPU) data transfer
    mem_height = 10 #size of datastructure in GPU. Keep unchanged
    #array size = mem_height x num_par
    num_par = 1_000_000 #number of parallel simulations
    num_batches = 1  # number of batches of simulation. set to 1. Functionality incomplete
    #total number of simulations = num_par*num_batches

    # physical parameters
    t_init = 0  #initial time
    t_end  = 10  #final time
    grid_size  = 100000 # Number of grid points
    sigma = 1.0 #scaling factor for noise
    ## Initial Conditions
    y_init = torch.normal(mean=2.0, std=np.sqrt(2.0), size=(num_par,), device = torch.device('cuda'), dtype=torch.float32)
    
    assert(grid_size%mem_height==0) #For efficiency reasons keep grid_size a multiple of mem_height 
    
    torch.cuda.empty_cache()
    q1 = mp.Queue(maxsize=cpu_buffer_size)

    
    print("Creating Large Arrays....")
    ys_cpu_buffer = torch.zeros(cpu_buffer_size, mem_height, num_par, dtype = torch.float32, device = torch.device('cpu'), pin_memory=True)
    ys_cpu_buffer.share_memory_()
    ys = torch.zeros(mem_height, num_par, dtype = torch.float32, device = torch.device('cuda'))
    dW = torch.zeros(num_par, dtype = torch.float32, device = torch.device('cuda'))

    print(f"GPU Memory: {tensor_memory(ys):.2f} MB")
    print(f"CPU Memory: {tensor_memory(ys_cpu_buffer[0])*cpu_buffer_size:.2f} MB")

    if task == 'write_data':
        proc = [mp.Process(target = writer, args = (q1, ys_cpu_buffer,)) for _ in range(threads)]
        for item in proc: item.start()
    elif task == 'find_density':
        p_dict = mp.Manager().dict()
        proc = [mp.Process(target = calc_density, args = (q1, ys_cpu_buffer, p_dict)), ]
        for item in proc: item.start()
        while 'start' not in p_dict: time.sleep(0.1)
    else: 
        proc = [] 
    

    dt     = float(t_end - t_init) / grid_size
    p_scale = 50/grid_size
    # Loop
    print("Starting Simulation...")
    for batch in range(num_batches):
        print(f"Simulation Batch: {batch+1}/{num_batches}")
        t1 = time.perf_counter()
        ys[0][:] = y_init
        buffer_index = 0
        num_parts = 0

        for i in range(1, grid_size):
            
            #updating progress bar
            if i%int(grid_size/50) == 0:
                print("<"+"="*int(i*p_scale)+"_"*int((grid_size-i-1)*p_scale)+">", end='\r')
        
            ################# Milstein method #################
            y = ys[(i-1)%mem_height]
            torch.normal(out=dW, mean=0.0, std=np.sqrt(dt), size=(num_par,), device = torch.device('cuda'), dtype=torch.float32)
            ode_term = y - dV(y) * dt
            stochastic_term = sigma* dW + 0.5* sigma**2 * (dW**2 - dt)
            torch.add(ode_term, stochastic_term, out = ys[i%mem_height])
            #ys[i%mem_height].copy_(y - dV(y) * dt  + sigma* dW + 0.5* sigma**2 * (dW**2 - dt))
  
            ################# Data Transfer: GPU -> CPU #################
            if i%mem_height==0:
                ys_cpu_buffer[buffer_index].copy_(ys, non_blocking=True)
                if proc != []: q1.put((buffer_index, output_directory+f'batch_{batch}_part_{num_parts}.ds'))
                buffer_index = (buffer_index+1)%cpu_buffer_size
                num_parts+=1
                
    
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
    parameters = {'t_init': t_init, 't_end': t_end, 'grid_size': grid_size,
                   'dt': dt, 'sigma': sigma, 'threads': threads, 'num_par': num_par,
                  'cpu_buffer_size': cpu_buffer_size,  'mem_height': mem_height,
                  'num_batches': num_batches, 'num_parts': num_parts
                  }
    with open(output_directory + "parameters.txt", 'w') as file:
        json.dump(parameters, file)
    np.save(output_directory + 'p_x.npy', p_dict['p_x'].numpy())
    np.save(output_directory + 'x.npy', p_dict['x'].numpy())

    t2 = time.perf_counter()
    print(f"Finished: {t2-t1:.2f}s")