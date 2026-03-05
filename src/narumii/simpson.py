import os
import numpy as np
import time
import subprocess
import warnings
from dataclasses import dataclass, field
from scipy.optimize import minimize_scalar
from narumii.dipolar import CTDrenar, Redor
from narumii.functions import compute_rmsd

# packages for type-hint
from typing import Any, Callable
from scipy.optimize import OptimizeResult


def create_filenames(
        basename: str, 
        template: str | None = None,
        **kwargs: dict[str, str]
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
    filenames = {}
    filenames['input'] = basename + '.in'
    filenames['output'] = basename + '.fid'
    filenames['log'] = basename + '.log'

    filenames['template'] = basename + '.template' if template is None else template

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
    params: dict[str, str] = field(init=False)

    def __post_init__(self):
        self.params = dict()
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
            self.params['mas'] = str(spin_rate*1000)

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

    def __post_init__(self):
        self.param_name = 'beff'
        self.residual_function = compute_rmsd

        # Default data extractor using CTDrenar
        if self.exp_type.lower() in ['ctdrenar', 'ct_drenar', 'ct-drenar']:
            self.data_extractor = lambda filepath: CTDrenar(filepath).difference
        elif self.exp_type.lower() in ['redor', 'reapdor']:
            self.data_extractor = lambda filepath: Redor(filepath).difference
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
        
        if not hasattr(self, 'data_extractor'):
            raise ValueError("Data extractor not provided.")
        
        # Run simulation with the given parameter value
        create_input_file(self.filenames, self.params)
        subprocess.run([self.simpson_path, self.filenames['input']], check=True)
        
        # Calculate residual using the data extractor
        simulated_data = self.data_extractor(self.filenames['output'])
        reference_data = self.data_extractor(self.filenames['reference'])
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
        residual_function: Callable | None = None, 
        bounds: tuple = (-1500, 0), 
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