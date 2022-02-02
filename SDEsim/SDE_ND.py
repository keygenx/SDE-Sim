from .SDE import *

class SDE_ND(SDE):
    """A class for simulating Autonomous SDEs using the Ito Scheme.
        The following arguments are possilbe:
        > num_par: int/long, number of parallel simulations to run.
        > dims: int/long, number of dimensions.
        > method: string, available methods: 'euler_maruyama', 'milstein' (default), 'taylor_15'
        > dtype: string, available dtypes: 'float32', 'float64' (default)
        > device: string, available devices: 'cpu', 'cuda' (default)
        > adaptive_stepping: bool, True (default) turns on adaptive time steps.
        > tolerance: float, local error tolerance for adaptive stepping. Default value = 1e-3
        > dt_init: float, initial step size. Deafult value = 1e-10
        """
    def __init__(self, num_par : int, dims: int, **kwargs):
        super().__init__(num_par, **kwargs)
        self.dims = dims
        self.y_init = torch.zeros((dims, num_par), dtype = self.dtype, device = self.device)
        
    def set_stats(self, **kwargs):
        
        for stat in ['info_rate', 'density', 'entropy', 'density_v', 'entropy_v', 'mean_v', 'std_v', 'info_rate_v']:
            if stat in kwargs:
                if kwargs[stat] == True: raise Exception(f"{stat} not implemented.")
            
        super().set_stats(**kwargs)

    
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
            self._step_forward_uncompiled = self._euler_maruyama
        elif self.method == 'milstein':
            self._step_forward_uncompiled = self._milstein
        elif self.method == 'taylor_15':     
            self._step_forward_uncompiled = self._taylor_15
        
        ####Defining forward step function
        ####First function definition is written as string and then compiled with exec and then compiled as torchscript
        ####Convoluted logic is used to make the torchscript compiled code efficient.
        dict1 = {'self': self, 'torch': torch}
        str_fn = "def func1(y, dw, dt):"
        for i in range(self.dims):
            fn_set = ", ".join([f"{fn[0]}y[{fn[2:]}]" if isinstance(fn, str) else f"self.fn[{i}][{j}]" for j, fn in enumerate(self.fn[i])])
            str_fn += f"\n  temp_{i} = self._step_forward_uncompiled(y[{i}], dw[{i}], dt, ({fn_set}))"
        str_fn += "\n  return torch.stack([" + ", ".join([f"temp_{i}" for i in range(self.dims)]) +"])"
        str_fn += "\nfunc1.__module__ = 'runtime_defined'"
        str_fn += "\nself.combine_step = func1"
        if self.debug:
            print("\nStep Forward Function Definition:")
            print(str_fn)
        exec(str_fn, dict1, dict1)
        self.step_forward = torch.jit.trace(self.combine_step, (self.y_init, self.y_init, self.dt_init))
        
        
        if self.adaptive_stepping == False: 
            self._functions_compiled = True
            return
    
        ####Defining step error function
        ####First function definition is written as string and then compiled with exec and then compiled as torchscript
        ####Convoluted logic is used to make the torchscript compiled code efficient.
        str_fn = "def func2(y_new, y_old, dW, dt):"
        for i in range(self.dims):
            str_fn += f"\n  z = torch.normal(mean=0.0, std=1.0, size=({self.num_par},), device = self.device, dtype={self.dtype})*dt**0.5"
            str_fn += f"\n  dW_1 = 0.5*dW[{i}] + 0.5*z"
            str_fn += f"\n  dW_2_{i} = 0.5*dW[{i}] - 0.5*z"
            fn_set = ", ".join([f"{fn[0]}y_old[{fn[2:]}]" if isinstance(fn, str) else f"self.fn[{i}][{j}]" for j, fn in enumerate(self.fn[i])])
            str_fn += f"\n  ys_1_{i} = self._step_forward_uncompiled(y_old[{i}], dW_1, dt/2, ({fn_set}))"
        for i in range(self.dims):
            fn_set = ", ".join([f"{fn[0]}ys_1_{fn[2:]}" if isinstance(fn, str) else f"self.fn[{i}][{j}]" for j, fn in enumerate(self.fn[i])])
            str_fn += f"\n  ys_2_{i} = self._step_forward_uncompiled(ys_1_{i}, dW_2_{i}, dt/2, ({fn_set}))"
            
        str_fn += "\n  return ((" + " + ".join([f"(ys_2_{i} - y_new[{i}])**2" for i in range(self.dims)]) + ")**0.5).max()/self._tolerance"
        str_fn += "\nfunc2.__module__ = 'runtime_defined'"
        str_fn += "\nself.combine_error = func2"
        exec(str_fn, dict1, dict1)
        if self.debug:
            print("\nStep Error Function Definition:")
            print(str_fn)
        self.step_error = torch.jit.trace(self.combine_error, (self.y_init, self.y_init, self.y_init, self.dt_init), check_trace=False)
        
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
                elif isinstance(fn[i], str):
                    temp[fn_str] = f"{fn_str} = {fn[i]}"
                else:
                    temp[fn_str] = inspect.getsource(fn[i])