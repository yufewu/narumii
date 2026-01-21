import os
import numpy as np
import time
import subprocess
import warnings
from scipy.optimize import minimize_scalar
from narumii.dipolar import CTDrenar
from narumii.functions import compute_rmsd


def create_filenames(
        basename: str, 
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
    filenames['template'] = basename + '.template'
    filenames['input'] = basename + '.in'
    filenames['output'] = basename + '.fid'
    filenames['log'] = basename + '.log'

    for key, value in kwargs.items():
        filenames[key] = value
    return filenames


def create_input_file(
    filenames: dict[str, str], 
    params: dict[str, str | float] | None = None, 
    angle_set: list[float] | None = None, 
    b: float | None = None
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
    exp_params: dictionary
        Container for experimental parameters.
    b: float
        Dipolar coupling constant. 
    angle_set: array_like
        Euler angles as an array: [alpha1, beta1, gamma1, alpha2, beta2, gamma2].

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

    # Replace parameters
    for key, value in params.items():
        if value is not None:
            content = content.replace(f"VALUE_{key.upper()}", str(value))
    
    # Replace angles
    for i in range(6):
        content = content.replace(f"VALUE_ANGLE{i+1}", str(angle_set[i]))

    # Replace other optional parameters
    if b is not None:
        content = content.replace("VALUE_B", str(b))

    content = content.replace("OUTPUT_FILE", filenames['output'])

    with open(filenames['input'], "w") as f:
        f.write(content)


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
    exp_params: dict
        Experimental parameters for the simulation.
    b: float
        Dipolar coupling constant.
    """
    def __init__(
        self, 
        simpson_path: str
        ) -> None:

        self.simpson_path = simpson_path
        self.angle_set = [0]*6


    def create_input_file(
        self, 
        filenames: dict[str, str] | None = None,
        exp_params: dict[str, str | int | float] | None = None, 
        b: float | None = None, 
        angle_set: list[float] | None = None, 
        **kwargs: dict[str, str | int | float]
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
        exp_params: dictionary
            Container for experimental parameters.
        b: float
            Dipolar coupling constant. 
        angle_set: array_like
            Euler angles as an array: [alpha1, beta1, gamma1, alpha2, beta2, gamma2].

        Returns
        ------
        None
        """

        if filenames is not None:
            self.filenames = filenames
        elif not hasattr(self, 'filenames'):
            raise ValueError("Filenames not provided.")

        with open(self.filenames['template']) as f:
            content = f.read()

        # set experimental parameters
        if exp_params is not None:
            for key, value in exp_params.items():
                self.exp_params[key] = value
        for key, value in self.exp_params.items():
            if value is not None:
                content = content.replace(f"VALUE_{key.upper()}", str(value))

        # set CS Euler angles
        if angle_set is not None:
            self.angle_set = angle_set
        for i in range(6):
            content = content.replace(f"VALUE_ANGLE{i+1}", str(self.angle_set[i]))

        # set b_eff value
        if b is not None:
            self.b = b
        content = content.replace("VALUE_BEFF", str(self.b))

        for key, value in kwargs.items():
            content = content.replace(f"VALUE_{key.upper()}", str(value))

        # set output file name
        content = content.replace("VALUE_OUTPUT_FILE", self.filenames['output'])

        with open(self.filenames['input'], "w") as f:
            f.write(content)


    def simulate(
        self, 
        filenames: dict[str, str] | None = None, 
        exp_params: dict[str, str | int | float] | None = None, 
        b: float | None = None, 
        angle_set: list[float] | None = None
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
        
        if exp_params is not None:
            self.exp_params = exp_params
        
        if b is not None:
            self.b = b
        
        if angle_set is not None:
            self.angle_set = angle_set

        tic = time.time()
        self.create_input_file()
        subprocess.run([self.simpson_path, self.filenames['input']], check=True)
        toc = time.time()
        elapsed_time = toc - tic

        return elapsed_time
    

class Optimizer(Simulator):
    """
    Optimizer class to optimize parameters in SIMPSON simulations.
    
    Inherits from Simulator class and provides methods for parameter optimization.
    Currently optimizes the b (dipolar coupling) parameter.
    """
    
    def __init__(
        self, 
        simpson_path: str, 
        exp_type: str = 'CTDrenar'
        ) -> None:

        super().__init__(simpson_path)
        self.param_name = 'beff'
        self.residual_function = compute_rmsd

        # Default data extractor using CTDrenar
        if exp_type.lower() in ['ctdrenar', 'ct_drenar', 'ct-drenar']:
            self.data_extractor = lambda filepath: CTDrenar(filepath).difference
        else:
            raise ValueError(f"Unsupported experiment type: {exp_type}")
    
    
    def objective(
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
        self.create_input_file(**{self.param_name: param})
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
        param_name: str = None, 
        residual_function: callable = None, 
        bounds: tuple = (-1500, 0), 
        method: str = 'bounded', 
        options: dict = {'xatol': 1}
        ) -> list[float, float, float]:
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
        
        if param_name is not None:
            self.param_name = param_name

        if residual_function is not None:
            self.residual_function = residual_function
        else:
            warnings.warn("No residual function provided, using default function (rmsd).")

        print(f"\n=== Optimizing===")
        tic = time.time()
        optimization_result = minimize_scalar(
            self.objective,
            bounds=bounds,
            method=method,
            options=options
        )
        toc = time.time()
        elapsed_time = toc - tic  # in seconds
        
        # Clean up temporary files
        if os.path.exists(self.filenames['input']):
            os.remove(self.filenames['input'])
        if os.path.exists(self.filenames['output']):
            os.remove(self.filenames['output'])
        
        return [
            optimization_result.x,
            optimization_result.fun,
            elapsed_time
        ]