import os
import sys
import time
import subprocess
import warnings
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import minimize_scalar
from .dipolar import CTDrenar, Redor, Redor3, DoubleQuantum
from .functions import compute_rmsd
from .utils import Tee

# packages for type-hint
from typing import Any, Callable, Literal, Sequence
from scipy.optimize import OptimizeResult


def create_filenames(
        basename: str, 
        template: str | None = None,
        **kwargs: str, 
    ) -> dict[str, str]:
    """
    Create filenames used for simulations based on a given basename. 
    
    Parameters
    -----
    basename: str
        Base name for simulation files.
    **kwargs: str
        Additional filenames to include in the returned dictionary.

    Returns
    ------
    filenames: dictionary
        Container for filenames, with keys: 'template', 'input', 'output', 'log', etc..
    """
    filenames = {
        'input': basename + '.in', 
        'output': basename + '.fid', 
        'log': basename + '.log', 
        'txt': basename + '.txt',
        'template': basename + '.template' if template is None else template, 
    }

    for key, value in kwargs.items():
        filenames[key] = value
    return filenames


def create_input_file(
    filenames: dict[str, str], 
    params: dict[str, str] | None = None
    ) -> None:
    """
    Create SIMPSON input file.

    Parameters
    -----
    filenames: dictionary
        Container for filenames, must have keys: 'template', 'input', 'output'.
        template: the template file based on which the input file is created. 
        input: the path of the created input file.
        output: the path of theoutput file to be created by simulation.
    params: dictionary
        Container for simulation parameters.
    Returns
    ------
    None
    """
    if 'input' not in filenames:
        warnings.warn("No input filename provided, using default 'input.in'.")
        filenames['input'] = 'input.in'
    
    if 'output' not in filenames:
        warnings.warn("No output filename provided, using default 'output.fid'.")
        filenames['output'] = 'output.fid'

    with open(filenames['template']) as f:
        content = f.read()

    params = dict() if params is None else params
    params['output_file'] = filenames['output']
        
    for key, value in params.items():
        if value is not None:
            content = content.replace(f"VALUE_{key.upper()}", str(value))

    with open(filenames['input'], "w") as f:
        f.write(content)

@dataclass
class Simulator:
    """
    Simulator class to run SIMPSON simulations.
    
    Parameters
    -----
    simpson_path: str
        Path to the SIMPSON executable.

    Attributes
    -----
    simpson_path: str
        Path to the SIMPSON executable.
    angle_set: list
        Euler angles as a list: [alpha1, beta1, gamma1, alpha2, beta2, gamma2].
    params: dict
        Experimental parameters for the simulation.
    b: float
        Dipolar coupling constant.
    """

    simpson_path: str
    params: dict[str, str] = field(default_factory=dict)
    

    def __post_init__(self):
        for i in range(6):
            self.params[f"angle{i+1}"] = '0'


    def simulate(
        self, 
        filenames: dict[str, str] | None = None, 
        b: float | None = None, 
        spin_rate: float | None = None,
        angle_set: list[float] | None = None, 
        **kwargs: dict[str, str]
        ) -> float:
        """
        Run SIMPSON simulation with the given input file.
        
        Parameters
        -----
        input_file: str
            Path to the SIMPSON input file.

        Returns
        ------
        elapsed_time: float
            Elapsed time for the simulation in seconds.
        """
        if filenames is not None:
            self.filenames = filenames
        elif not hasattr(self, 'filenames'):
            raise ValueError("Filenames not provided.")
        
        if b is not None:
            self.b = b
            self.params['beff'] = str(b)
        
        if spin_rate is not None:
            self.spin_rate = spin_rate
            self.params['spin_rate'] = str(spin_rate)

        if angle_set is not None:
            self.angle_set = angle_set
            for i in range(6):
                self.params[f"angle{i+1}"] = str(angle_set[i])
        
        for key, value in kwargs.items():
            self.params[key] = str(value)

        tic = time.time()
        create_input_file(self.filenames, self.params) 
        subprocess.run([self.simpson_path, self.filenames['input']], check=True)
        toc = time.time()
        elapsed_time = toc - tic

        return elapsed_time
    

@dataclass
class Optimizer(Simulator):
    """
    Optimizer class to optimize parameters in SIMPSON simulations.
    
    Inherits from Simulator class and provides methods for parameter optimization.
    Currently optimizes the b (dipolar coupling) parameter.
    """
    exp_type: str = 'CTDrenar'

    param_name: str = field(init=False)
    residual_function: Callable = field(init=False)

    def __post_init__(self):
        for i in range(6):
            self.params[f"angle{i+1}"] = '0'

        # Default data extractor using CTDrenar
        if self.exp_type.lower() in ['ctdrenar', 'ct_drenar', 'ct-drenar']:
            self.data_parser = lambda filepath: CTDrenar(filepath).difference
        elif self.exp_type.lower() in ['redor', 'reapdor']:
            self.data_parser = lambda filepath: Redor(filepath).difference
        elif self.exp_type.lower() in ['redor3']:
            self.data_parser = lambda filepath: Redor3(filepath).difference
        elif self.exp_type.lower() in ['dq', 'doublequantum', 'double-quantum', 'double_quantum']:
            self.data_parser = lambda filepath: DoubleQuantum(filepath).difference
        else:
            raise ValueError(f"Unsupported experiment type: {self.exp_type}")
    
    
    def _objective(
        self, 
        param: float
        ) -> float:
        """
        Objective function for optimization. Evaluates residual for a given parameter value.
        
        Parameters
        -----
        param: float
            The parameter value to evaluate.
        
        Returns
        ------
        residual: float
            The residual between simulated and reference data.
        """
        if not hasattr(self, 'residual_function') or self.residual_function is None:
            raise ValueError("Residual function not provided.")
        
        if not hasattr(self, 'data_parser'):
            raise ValueError("Data extractor not provided.")
        
        # Run simulation with the given parameter value
        self.params[self.param_name] = str(param)
        create_input_file(self.filenames, self.params)
        subprocess.run([self.simpson_path, self.filenames['input']], check=True)
        
        # Calculate residual using the data extractor
        simulated_data = self.data_parser(self.filenames['output'])
        reference_data = self.data_parser(self.filenames['reference'])
        residual = self.residual_function(simulated_data, reference_data)
        print(f"  {self.param_name} = {param:.2f} Hz, residual = {residual:.4f}")

        # Log the result if log file is specified
        if 'log' in self.filenames:
            write_header = not os.path.exists(self.filenames['log'])
            with open(self.filenames['log'], "a") as f:
                if write_header:
                    f.write(f"{self.param_name},residual\n")
                f.write(f"{param:.2f},{residual:.4f}\n")
        
        return residual
    

    def optimize(
        self, 
        param_name: str, 
        residual_function: Callable = residual_function, 
        bounds: Sequence = (-1500, 0), 
        method: str = 'bounded', 
        options: dict = {'xatol': 1}
        ) -> tuple[float, float, float]:
        """
        Optimize the specified parameter.
        
        Parameters
        -----
        param_name: str
            Name of the parameter to optimize (default: 'beff').
        residual_function: function
            Function to compute residuals.
        bounds: tuple
            (min, max) bounds for optimization.
        method: str
            Optimization method supported in scipy.optimize.minimize_scalar (default: 'bounded').
        options: dict
            Additional options for the optimizer (default: {'xatol': 1}).
        
        Returns
        ------
        optimization_result.x: float
            Optimized parameter value.
        optimization_result.fun: float
            Residual at optimized parameter.
        elapsed_time: float
            Elapsed time for the optimization in seconds.
        """   
        if not hasattr(self, 'filenames'):
            raise ValueError("Filenames not provided.")
        
        if param_name.lower() == 'b':
            param_name = 'beff'
        self.param_name = param_name

        if residual_function is not None:
            self.residual_function = residual_function
        else:
            warnings.warn("No residual function provided, using default function (rmsd).")
            self.residual_function = compute_rmsd

        print(f"\n=== Optimizing===")
        tic = time.time()
        optimization_result: OptimizeResult = minimize_scalar(
            self._objective,
            bounds=bounds,
            method=method,
            options=options
        )   # type: ignore
        toc = time.time()
        elapsed_time = toc - tic  # in seconds
        
        # Clean up temporary files
        if os.path.exists(self.filenames['input']):
            os.remove(self.filenames['input'])
        if os.path.exists(self.filenames['output']):
            os.remove(self.filenames['output'])
        
        return optimization_result.x, optimization_result.fun, elapsed_time


def run_simulation(
    simpson_path: str,
    basename: str = './new_simulation',
    template: str | None = None,
    beff: float | None = None,
    params: list[str] | dict[str, str] = [],
    verbose: bool = True
) -> float:
    """
    Run a single Simpson simulation using an input template and provided parameters.

    Parameters
    ----------
    simpson_path : str
        Path to the Simpson executable.
    basename : str, optional
        Basename of the simulation file. Files will be generated based on this basename:
        - input - basename.in
        - output - basename.fid
        Default is './new_simulation'. 
    template : str, optional
        Path to the input template file used to generate the real input file from provided parameters. If None, 
        the filename basename.template will be used. Default is None. 
    beff : float, optional
        Dipolar coupling constant in Hz. Should have the opposite sign to the product of the gyromagnetic ratios of both nuclei. 
        Default is None.
    params : list of str or dict, optional
        Simulation parameters. Can be either:
        - A list of strings in 'key=value' format, e.g., ['spin_rate=17000.0', 'l0=1']
        - A dictionary of key-value pairs, e.g., {'spin_rate': '17000.0', 'l0': '1'}
        Default is None.
    verbose : bool, optional
        If True, print simulation progress to stdout. Default is True.

    Returns
    -------
    elapsed_time: float
        Elapsed time for the simulation in seconds.

    Examples
    --------
    >>> run_simulation(simpson_path, basename='sim', template='input_template.txt', beff=-600)
    >>> run_simulation(simpson_path, basename='sim', template='input_template.txt', beff=-600, 
    ...                params=['spin_rate=17000.0', 'l0=1'])
    """
    simulator = Simulator(simpson_path)
    simulator.filenames = create_filenames(basename, template=template)
    
    if isinstance(params, list):
        for param in params:
            if '=' not in param:
                warnings.warn(f"Ignoring invalid parameter format '{param}'. Please use 'key=value' format.")
                continue
            key, value = param.split('=', 1)
            simulator.params[key] = value
    elif isinstance(params, dict):
        simulator.params.update(params)
    
    print(f"\n=== Simulating ===") if verbose else ""
    elapsed_time = simulator.simulate(b=beff)
    print(f"Simulation done! Elapsed time: {elapsed_time/60:.1f} min.")  if verbose else ""
    
    return elapsed_time


def run_optimization(
    simpson_path: str,
    exp_type: str,
    basename: str = './new_optimization',
    template: str | None = None,
    reference: str = './reference.txt',
    params: list[str] | dict[str, str] | None = None,
    param_name: str = 'b', 
    residual_function: Callable = compute_rmsd, 
    method: str = "bounded", 
    bounds: Sequence = (-1500, 0), 
    options = {'xatol': 1}, 
    verbose: bool = True, 
) -> tuple[float, float, float]:
    """
    Optimize a parameter using SIMPSON simulations. Runs by iteratively running simulations and comparing the results 
    with a reference experimental data.

    Parameters
    ----------
    simpson_path : str
        Path to the Simpson executable.
    exp_type : str
        Type of NMR experiment. Supported values (either in full capitalized or small forms):  
        - 'CTDRENAR', 'CT-DRENAR' for CTDrenar experiments  
        - 'REDOR', 'REAPDOR' for REDOR/REAPDOR experiments  
        - 'REDOR3' for compensated REDOR experiments  
        - 'DQ', 'DOUBLEQUANTUM', 'DOUBLE-QUANTUM' for double quantum spectroscopy  
    basename : str, optional
        Basename of the simulation file. Files will be generated based on this basename:  
        - input - basename.in  
        - output - basename.fid  
        - log (as number-only csv) - basename.log  
        - log (more readable with rich information) - basename.txt  
        Default is './new_optimization'. 
    template : str, optional
        Path to the input template file used to generate the real input file from provided parameters. If None, 
        the filename basename.template will be used. Default is None. 
    reference : str, optional
        Path to the reference experimental data file. Default is './reference.txt'.
    params : list[str] or dict, optional
        Additional simulation parameters. Can be either:  
        - A list of strings in 'key=value' format  
        - A dictionary of key-value pairs  
        Default is None.
    param_name: str, optional
        Name of the parameter to be optimized. Default is 'b'. 
    residual_function: Callable, optional
        The function used to calculate the residual difference between simulation and experiment. Default is .functions.compute_rmsd.  
    method: str, optional
        Optimization method used by scipy.optimize. Currently only the method 'bounded' is implemented. 
        See [scipy document](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html). 
        Default is "bounded". 
    bounds: Sequence, optional
        Optimization bounds. Must have two finite items. 
        See [scipy document](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html). 
        Default is "bounded". (-1500, 0). 
    options: dict, optional
        A dictionary of additional solver options.
        See [scipy document](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html). 
        Default is {'xatol': 1}. 
    verbose : bool, optional
        If True, print optimization progress to stdout. Default is True.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing:  
        - b_opt : float  
            Optimized dipolar coupling constant value in Hz.  
        - residual : float  
            Residual (RMSD) at the optimized b_eff value.  
        - elapsed_time : float  
            Total elapsed time for optimization in seconds.

    Examples
    --------
    >>> b_opt, residual, time = run_optimization(
    ...     simpson_path,
    ...     exp_type='CT-DRENAR',
    ...     basename='dataset_name_opt',
    ...     reference='dataset_name_int.txt', 
    ... )
    >>> b_opt, residual, time = run_optimization(
    ...     simpson_path,
    ...     exp_type='CT-DRENAR',
    ...     basename='dataset_name_opt',
    ...     template='input_template.txt',
    ...     reference='dataset_name_int.txt', 
    ...     bounds=(-3000, 0), 
    ... )
    """
    if method != 'bounded': 
        raise NotImplementedError("Only 'bounded' method is implemented at the moment. ")
    if params is None:
        params = []
        
    optimizer = Optimizer(simpson_path, exp_type=exp_type)
    optimizer.filenames = create_filenames(basename, template=template, reference=reference)
    
    if isinstance(params, list):
        for param in params:
            if '=' not in param:
                warnings.warn(f"Ignoring invalid parameter format '{param}'. Use key=value format.")
                continue
            key, value = param.split('=', 1)
            optimizer.params[key] = value
    elif isinstance(params, dict):
        optimizer.params.update(params)

    log_filename = f"{basename}.txt"
    original_stdout = sys.stdout
    
    with open(log_filename, 'w', encoding='utf-8') as f:
        sys.stdout = Tee(sys.stdout, f) if verbose else Tee(f)
        
        try:
            result = optimizer.optimize(
                param_name=param_name, 
                residual_function=residual_function, 
                bounds=bounds, 
                method=method, 
                options=options
            )
            print(f"Best b_eff = {result[0]:.2f} Hz, residual = {result[1]:.4f}, elapsed time: {result[2]/60:.1f} min. ")
            return result
        finally:
            sys.stdout = original_stdout