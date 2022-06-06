from .SDE import *
import re 

class SDE_ND(SDE):
    __doc__ = SDE.__doc__ + '> dims: int/long, number of dimensions.'
    def __init__(self, num_par : int, dims: int, **kwargs):
        super().__init__(num_par, **kwargs)
        self.dims = dims
        self.y_init = torch.zeros((dims, num_par), dtype = self.dtype, device = self.device)
        self._shape = (dims, num_par)

    def set_stats(self, **kwargs):
        
        for stat in ['density', 'entropy', 'density_v', 'entropy_v']:
            if stat in kwargs:
                print(f"Warning: Only statistics for unconditional probability along each dimension can be measured with this implementation.",
                "Stats for joint probability will be implemented in Future releases.")
                break

        for stat in ['info_rate_v','info_rate']:
            if stat in kwargs:
                if kwargs[stat] == True: raise Exception(f"{stat} not implemented.")
            
        super().set_stats(**kwargs)
        
    #function to make life easier in the following code.
    def get_fn_set(self, i, replace, find = 'y([0-9]*)', new_t = None):
        temp = []
        for j, fn in enumerate(self.fn[i]):
            if isinstance(fn, str):
                string = re.sub(find, replace, fn)
                if new_t != None: 
                    #using regex lookahead, lookbehind, character class, capturing group
                    #find standalone t and replace it with new_t
                    string = re.sub('(?<![a-zA-Z0-9_])t(?![a-zA-Z0-9_])', new_t, string)
                temp.append(string)
            elif callable(fn):
                temp.append(f"self.fn[{i}][{j}]")
            else:
                temp.append(f"{fn}")
        return ', '.join(temp)
            
    def step_forward_generate(self):
        ####Defining forward step function
        ####First function definition is written as string and then compiled with exec and later compiled as torchscript
        ####Convoluted logic is used to make the torchscript compiled code efficient.
        dict1 = {'self': self, 'torch': torch}
        str_fn = "def func(y, dw, dt, t):"
        for i in range(self.dims):
            fn_set = self.get_fn_set(i, 'y[\\1]')
            str_fn += f"\n  temp_{i} = self._step_fn(y[{i}], dw[{i}], dt, ({fn_set}))"
        str_fn += "\n  return torch.stack([" + ", ".join([f"temp_{i}" for i in range(self.dims)]) +"])"
        str_fn += "\nfunc.__module__ = 'runtime_defined'"
        str_fn += "\nself._step_forward_uncompiled = func"
        if self.debug:
            print("\nStep Forward Function Definition:")
            print(str_fn, end = '\n\n')
        exec(str_fn, dict1, dict1)
        
    def step_error_generate(self):
        ####Defining step error function
        ####First function definition is written as string and then compiled with exec and then compiled as torchscript
        ####Convoluted logic is used to make the torchscript compiled code efficient.
        dict1 = {'self': self, 'torch': torch}
        str_fn = "def func2(y_new, y_old, dW, dt, t):"
        for i in range(self.dims):
            str_fn += f"\n  z = torch.normal(mean=0.0, std=1.0, size=({self.num_par},), device = dW.device, dtype=dW.dtype)*dt**0.5"
            str_fn += f"\n  dW_1 = 0.5*dW[{i}] + 0.5*z"
            str_fn += f"\n  dW_2_{i} = 0.5*dW[{i}] - 0.5*z"
            fn_set = self.get_fn_set(i, 'y_old[\\1]')
            str_fn += f"\n  ys_1_{i} = self._step_fn(y_old[{i}], dW_1, dt/2, ({fn_set}))"
        for i in range(self.dims):
            fn_set = self.get_fn_set(i, 'ys_1_\\1', new_t = '(t+dt/2)')
            str_fn += f"\n  ys_2_{i} = self._step_fn(ys_1_{i}, dW_2_{i}, dt/2, ({fn_set}))"
        #str_fn += "\n  print('Sub:', (torch.abs(ys_2_0 - y_new[0]).max()/self._tolerance), (torch.abs(ys_2_1 - y_new[1]).max()/self._tolerance), (torch.abs(ys_2_2 - y_new[2]).max()/self._tolerance).item())"
        #str_fn += "\n  print('Tot: ', ((((ys_2_0 - y_new[0])**2 + (ys_2_1 - y_new[1])**2 + (ys_2_2 - y_new[2])**2)**0.5).max()/self._tolerance).item())"
        str_fn += "\n  return ((" + " + ".join([f"(ys_2_{i} - y_new[{i}])**2" for i in range(self.dims)]) + ")**0.5).max()/self._tolerance"
        str_fn += f"\nfunc2.__module__ = 'runtime_defined'"
        str_fn += "\nself.step_error_uncompiled = func2"
        
        if self.debug:
            print("Step Error Function Definition:")
            print(str_fn, end='\n\n')
        exec(str_fn, dict1, dict1)
        
    
    def set_functions(self, *fn_sets):
        if len(fn_sets) != self.dims:
            raise Exception("Number of set of functions doesn't match Dimensions = {self.dims}")
            
        if len(fn_sets) != self.dims:
            raise Exception("Number of function sets provided don't match diemnsion")
        
        self.fn = []
        ####Populating self.fn -> list of functions
        for i,fn_set in enumerate(fn_sets):
            if len(fn_set)<2:
                raise Exception(f"Dim {i}: Atleast drift and diffusion functions should be provided")
            temp_list = []
            for j,fn in enumerate(fn_set):
                temp_list.append(fn)
            for j in range(len(fn_set), len(self.fn_def_strings)):
                temp_list.append(None)
            self.fn.append(temp_list)
        print
        ####Choosing stepping algorithm
        if self.method == 'euler_maruyama':
            self._step_fn = self._euler_maruyama
        elif self.method == 'milstein':
            self._step_fn = self._milstein
        elif self.method == 'taylor_15':     
            self._step_fn = self._taylor_15
        
        self.step_forward_generate()
        self.step_forward = torch.jit.trace(self._step_forward_uncompiled, (self.y_init, self.y_init, self.dt_init, self.dt_init))
        
        self.velocity = torch.jit.trace(self._velocity, (self.y_init, self.dt_init), check_trace = False)
        
        if self.adaptive_stepping: 
            self.step_error_generate()
            self.step_error = torch.jit.trace(self.step_error_uncompiled, (self.y_init, self.y_init, self.y_init, self.dt_init, self.dt_init), check_trace=False)
        
        self._functions_compiled = True

    def _initialize_y(self, init_mean, init_std):
        size = (self.dims, self.num_par)
        ones = torch.ones(size = size, device=self.device, dtype=self.dtype)

        if isinstance(init_mean, (int, float)):
            mean = ones*init_mean
        elif len(init_mean) == self.dims:
            mean = ones*torch.tensor(init_mean, device=self.device, dtype=self.dtype)[:, None]
        else:
            raise Exception("Length of init_mean argument doesn't match dimensions")

        if isinstance(init_std, (int, float)):
            std = ones*init_std
        elif len(init_std) == self.dims:
            std = ones*torch.tensor(init_std, device=self.device, dtype=self.dtype)[:, None]
        else:
            raise Exception("Length of init_std argument doesn't match dimensions")
            
        return torch.normal(mean = mean, std = std)

    def simulate(self, t_init : float, t_end : float, init_mean, init_std):
        """
        > t_init : float, inital time 
        > t_end : float, final time
        > init_mean : float/tuple/array/tensor, mean of initial distribution
        > init_std : float/tuple/array/tenosr, standard deviation of initial distribution
        
        For multi-dimensional simulation if float is passed for either init_std or init_mean, all dimension will be 
        taken to have the same init_std or init_mean.
        """
        super().simulate(t_init, t_end, init_mean, init_std)

    def _save_functions(self):
        self.data['parameters']['dims'] = self.dims
        self.data['functions'] = {}
        for j, fn in enumerate(self.fn):
            temp = self.data['functions'][f"dim:{j}"] = {}
            for i, fn_str in enumerate(self.fn_def_strings):
                if fn[i] == None:
                    temp[fn_str] = 'None'
                elif callable(fn[i]):
                    temp[fn_str] = inspect.getsource(fn[i])
                else:
                    temp[fn_str] = fn[i]