import torch
import numpy as np
import time
import json

def tensor_memory(obj):
    """Returns memory size of a torch.tensor in MB"""
    return obj.element_size()*obj.nelement()/10**6

def hello_world():
    print("Hello World")

@torch.jit.script
def cuda_histogram(item, bins: int, density: bool = True):

    min_val = torch.min(item)
    max_val = torch.max(item)
    dx = (max_val-min_val)/bins

    bin_edge  = torch.linspace(min_val, max_val, bins+1, device = torch.device('cuda'))
    bin_centre = (bin_edge[1:]+bin_edge[:-1])/2

    prob = torch.histc(item, bins=bins, max=max_val, min =min_val )

    if density: prob = prob/(item.shape[0]*dx)
    return prob, bin_centre

def calc_density(q, ys_cpu_buffer, output_directory, started, n_bins = 400):
    print("Density Calculator Started")
    started.value = True
    t1 = time.perf_counter()
    time_list, prob_list, bin_list = [], [], []
    index, t_list = q.get()
    while index!= 'e':
        time_list.extend(t_list)
        tensor = ys_cpu_buffer[index][:len(t_list)]
        
        for item in tensor:
            prob_den, bin_edges = torch.histogram(item, n_bins, density = True)
            bin_centres = (bin_edges[:-1]+bin_edges[1:])/2
            prob_list.append(prob_den)
            bin_list.append(bin_centres)    
        index, t_list = q.get()  

    np.save(output_directory + 'p_x.npy', torch.stack(prob_list).numpy())
    np.save(output_directory + 'x.npy', torch.stack(bin_list).numpy())
    np.save(output_directory + 't.npy', np.stack(time_list))
    t2 = time.perf_counter()
    print(f"Density Calculator Finished: {t2-t1:.2f}s")
   
def calc_density(q, ys_cpu_buffer, output_directory, started, n_bins = 400):
    print("Density Calculator Started")
    started.value = True
    t1 = time.perf_counter()
    time_list, prob_list, bin_list = [], [], []
    index, t_list = q.get()
    while index!= 'e':
        time_list.extend(t_list)
        tensor = ys_cpu_buffer[index][:len(t_list)]
        
        for item in tensor:
            prob_den, bin_edges = torch.histogram(item, n_bins, density = True)
            bin_centres = (bin_edges[:-1]+bin_edges[1:])/2
            prob_list.append(prob_den)
            bin_list.append(bin_centres)    
        index, t_list = q.get()  

    np.save(output_directory + 'p_x.npy', torch.stack(prob_list).numpy())
    np.save(output_directory + 'x.npy', torch.stack(bin_list).numpy())
    np.save(output_directory + 't.npy', np.stack(time_list))
    t2 = time.perf_counter()
    print(f"Density Calculator Finished: {t2-t1:.2f}s")
       
def writer(q, ys_cpu_buffer):
    print("Writer Started")
    index, f_name = q.get()
    t1 = time.perf_counter()
    while f_name!='e':
        obj = ys_cpu_buffer[index]
        torch.save(obj, f_name)
        index, f_name = q.get()
    q.put(("e","e"))
    t2 = time.perf_counter()
    print(f"Writer Finished: {t2-t1:.2f}s")
    
def reader(q, directory):
    print("Reader Started")
    with open(directory + 'parameters.txt', 'r') as file:
        params = json.load(file)
    for part in range(100):
        print("Part", part)
        q.put(torch.cat([torch.load(directory + f'batch_{batch}_part_{part}.ds') for batch in range(params['num_batches'])], dim = 1))
    q.put('e')
    print("Reader finished")
    