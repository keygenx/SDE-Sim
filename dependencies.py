def tensor_memory(obj):
    return obj.element_size()*obj.nelement()/10**6

def hello_world():
    print("Hello World")

def derivative(tensor, dt):
    return (tensor[1:]-tensor[0:-1])/dt

def calc_density(q, ys_cpu_buffer, return_dict, n_bins = 400):
    import numpy as np
    import torch
    import time
    print("Density Calculator Started")
    return_dict['start'] = True
    t1 = time.perf_counter()
    index, _ = q.get()
    prob_list = []
    bin_list = []
    while index!= 'e':
        tensor = ys_cpu_buffer[index]
        for item in tensor:
            prob_den, bin_edges = torch.histogram(item, n_bins, density = True)
            bin_centres = (bin_edges[:-1]+bin_edges[1:])/2
            prob_list.append(prob_den)
            bin_list.append(bin_centres)
        index, _ = q.get()    
    return_dict['p_x'] = torch.stack(prob_list)
    return_dict['x'] = torch.stack(bin_list)
    t2 = time.perf_counter()
    print(f"Density Calculator Finished: {t2-t1:.2f}s")
    
def calc_density_depr(q, return_dict, n_bins = 400):
    import numpy as np
    import torch
    import time
    print("Density Calculator Started")
    t1 = time.perf_counter()
    tensor, _ = q.get()
    print("child:", id(tensor))
    print("child_sum:", torch.sum(tensor))
    prob_list = []
    bin_list = []
    while tensor!= 'e':
        for item in tensor:
            pass
            prob_den, bin_edges = np.histogram(item, n_bins, density = True)
            bin_centres = (bin_edges[:-1]+bin_edges[1:])/2
            prob_list.append(prob_den)
            bin_list.append(bin_centres)
        tensor, _ = q.get()
    return_dict['p_x'] = torch.tensor(prob_list)
    return_dict['x'] = torch.tensor(bin_list)
    t2 = time.perf_counter()
    print(f"Density Calculator Finished: {t2-t1:.2f}s")
    

def writer(q, ys_cpu_buffer):
    import torch
    import time
    print("Writer Started")
    index, f_name = q.get()
    t1 = time.perf_counter()
    while f_name!='e':
        obj = ys_cpu_buffer[index]
        torch.save(obj, f_name)
        index, f_name = q.get()
        num+=1
    q.put(("e","e"))
    t2 = time.perf_counter()
    print(f"Writer Finished: {t2-t1:.2f}s")
    
def reader(q, directory):
    import torch
    import json
    print("Reader Started")
    with open(directory + 'parameters.txt', 'r') as file:
        params = json.load(file)
    for part in range(100):
        print("Part", part)
        q.put(torch.cat([torch.load(directory + f'batch_{batch}_part_{part}.ds') for batch in range(params['num_batches'])], dim = 1))
    q.put('e')
    print("Reader finished")
    
def push_to_gpu(q_cpu, q_gpu):
    print("Push to GPU Started")
    data = q_cpu.get()
    i = 0
    print(i)
    while data!= 'e':
        print(i, q_cpu.qsize(), q_gpu.qsize())
        i+=1
        q_gpu.put(data.to('cuda', non_blocking = True))
        data = q_cpu.get()
    q_gpu.put('e')
    print("Push to GPU Ended")