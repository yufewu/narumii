"""
Useful python classes and functions for handling dipolar NMR experiments. 

Author(s): Yufei Wu, Julius Schlueter
Created on: 2025/08/08
Last modified: 2026/03/05
"""

from pathlib import PurePath
import warnings
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import pi, physical_constants
from dataclasses import dataclass
from .utils import read_fid
from .functions import ctdrenar, redor_bessel

# packages for type-hint
from typing import Any, Callable, Literal
from numbers import Number
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def _set_phases(
        phase_range: tuple[float, float] | None, 
        phase_increment: float | None, 
        n_points: int
        ) -> tuple[tuple[float, float], float]:
    """
    Calculate the phase-related parameters in CT-DRENAR experiments. 

    Parameters:
    - phase_range: tuple, (min_phase, max_phase) in degrees
    - phase_increment: float, increment of phase in degrees
    - n_points: int, number of points in the experiment

    Returns:
    - phase_range: tuple, (min_phase, max_phase) in degrees
    - phase_increment: float, increment of phase in degrees
    """
    if phase_increment is not None:
        if phase_range is not None and phase_increment != (phase_range[1] - phase_range[0]) / (n_points - 1):
            warnings.warn("Provided phase range does not match the provided increments and number of points. Please check")
        else: 
            phase_range = (0, phase_increment * (n_points - 1))
    else:
        if phase_range is None:
            warnings.warn("Using default phase range = 0-180 degrees. ")
            phase_range = (0, 180)
        phase_increment = (phase_range[1] - phase_range[0]) / (n_points - 1)

    return phase_range, phase_increment


def _set_x_axis(
        initial_value: float,
        loop_counter_start: int, 
        loop_counter_increment: int, 
        length_per_counter: float, 
        n_points: int, 
        num_continuous: int = 100
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    
    """
    loop_counters = np.linspace(loop_counter_start, 
                                 loop_counter_increment * (n_points - 1) + loop_counter_start, 
                                 num=n_points)
    x_discrete = loop_counters * length_per_counter + initial_value
    x_continuous = np.linspace(x_discrete[0], x_discrete[-1], num=num_continuous)

    return x_discrete, x_continuous, loop_counters


def _calculate_difference(modulated: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if np.any(reference == 0):
        raise ValueError("Reference contains zero(s), cannot divide.")
    return 1 - modulated / reference


def _to_fid(filename: str, 
           data: np.ndarray, 
           key: str = 'data',
           ) -> None:
    """
    Export data to a .fid file.

    Parameters:
    - filename: str, output file path
    - data: np.ndarray, data to be exported
    - key: str, attribute name of the data to be exported (default: 'data')

    Returns: 
    - None
    """
    
    with open(filename, 'w') as f:
        f.write('SIMP\n')
        f.write(f'NP={np.shape(data)[0]}\n')
        f.write('SW=100000\n')
        f.write('TYPE=FID\n')
        f.write('DATA\n')
        for value in data:
            f.write(f'{value} 0\n')
        f.write('END\n')


def _to_txt(filename: str, 
            data: np.ndarray, 
            order: Literal['C', 'F'] = 'C', 
            fmt: str ='%.6f', 
            ) -> None:
    """
    Export data to a .txt file.

    Parameters:
    - filename: str, output file path
    - data: np.ndarray, data to be exported
    - order: str, order of reshaping the data, 'C' for row-major, 'F' for column-major (default: 'C')
    - fmt: str, formatting for float values

    Returns:
    - None
    """
    data = data.reshape((-1, ), order=order)
    np.savetxt(filename, data, fmt=fmt)

@dataclass
class Fid_single:
    """
    Template calss for 1D intensity data with single-FID acquisition, such as CT-DRENAR. 
    """
    filename: str

    data: np.ndarray = field(init=False)
    modulated: np.ndarray = field(init=False)
    difference: np.ndarray = field(init=False)
    reference: np.ndarray = field(init=False)

    x_initial_value: float = 0
    loop_counter_start: int | None = None
    loop_counter_increment: int | None = None
    length_per_counter: float | None = None
    n_points: int | None = None
    num_continuous: int = 100

    reference_idx: int | None = None
    x_discrete: np.ndarray = field(init=False)
    x_continuous: np.ndarray = field(init=False)

    load_text_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialization: read data from .txt or .fid file. 

        Parameters:
        - filename: str, input file path
        - **kwargs: additional keyword arguments for np.loadtxt() when reading .txt file, see https://numpy.org/doc/2.1/reference/generated/numpy.loadtxt.html
        """
        if PurePath(self.filename).suffix == '.txt':
            self.data = np.loadtxt(self.filename, **self.load_text_options)
        elif PurePath(self.filename).suffix == '.fid':
            self.data, _ = read_fid(self.filename)
        else:
            raise TypeError("File must be .txt or .fid! ")
        
        if isinstance(self.loop_counter_start, Number) and isinstance(self.loop_counter_increment, Number) and isinstance(self.length_per_counter, Number) and isinstance(self.n_points, int):
            self.set_x_axis(self.x_initial_value, 
                            self.loop_counter_start, 
                            self.loop_counter_increment, 
                            self.length_per_counter, 
                            self.n_points, 
                            self.num_continuous)
        
        if isinstance(self.reference_idx, int):
            if self.reference_idx < 0 or self.reference_idx >= len(self.data):
                raise ValueError("Reference index is out of bounds.")
            self.calculate_difference(self.reference_idx)


    def set_x_axis(self, 
                x_initial_value: float,
                loop_counter_start: int, 
                loop_counter_increment: int, 
                length_per_counter: float, 
                n_points: int, 
                num_continuous: int,
                ) -> tuple[np.ndarray, np.ndarray]:
            self.x_initial_value = x_initial_value
            self.loop_counter_start = loop_counter_start
            self.loop_counter_increment = loop_counter_increment
            self.length_per_counter = length_per_counter
            self.n_points = n_points
            self.num_continuous = num_continuous

            self.x_discrete, self.x_continuous, _ = _set_x_axis(x_initial_value, 
                                                                loop_counter_start, 
                                                                loop_counter_increment, 
                                                                length_per_counter, 
                                                                n_points, 
                                                                num_continuous)
            return self.x_discrete, self.x_continuous


    def calculate_difference(self, reference_idx: int) -> np.ndarray:
        self.reference_idx = reference_idx

        if self.reference_idx < 0 or self.reference_idx >= len(self.data):
            raise ValueError("Reference index is out of bounds.")
        
        self.reference = np.ones(len(self.data)) * self.data[self.reference_idx]
        self.difference = _calculate_difference(self.data, self.reference)

        return self.difference


    def to_fid(self, 
               filename: str, 
               key: str = 'data',
               ) -> None:
      
        _to_fid(filename, getattr(self, key))
    

    def to_txt(self, 
               filename: str, 
               key: str = 'data',
               order: Literal['C', 'F'] = 'C',
               fmt: str ='%.6f', 
               ) -> None:

        _to_txt(filename, getattr(self, key), order=order, fmt=fmt)


@dataclass
class Fid_pair:
    """
    Template calss for 1D intensity data with double-FID acquisition, such as REDOR, VT-DRENAR, DQ build-up. 
    """
    filename: str

    data: np.ndarray = field(init=False)
    modulated: np.ndarray = field(init=False)
    difference: np.ndarray = field(init=False)
    reference: np.ndarray = field(init=False)

    x_initial_value: float = 0
    loop_counter_start: int | None = None
    loop_counter_increment: int | None = None
    length_per_counter: float | None = None
    n_points: int = field(init=False)
    num_continuous: int = 100

    x_discrete: np.ndarray = field(init=False)
    x_continuous: np.ndarray = field(init=False)
    loop_counters: np.ndarray = field(init=False)

    load_text_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialization: read data from .txt or .fid file. 

        Parameters:
        - filename: str, input file path
        - **kwargs: additional keyword arguments for np.loadtxt() when reading .txt file, see https://numpy.org/doc/2.1/reference/generated/numpy.loadtxt.html
        """
        if PurePath(self.filename).suffix == '.txt':
            self.data = np.loadtxt(self.filename, **self.load_text_options)
        elif PurePath(self.filename).suffix == '.fid':
            self.data, _ = read_fid(self.filename)
        else:
            raise TypeError("File must be .txt or .fid! ")
        
        if self.data.shape[0] % 2 == 1:
            raise ValueError("Data length is odd, cannot be reshaped into pairs.")
        self.data = np.reshape(self.data, (2, -1))
        self.n_points = self.data.shape[1]
        self.modulated = self.data[0]
        self.reference = self.data[1]
        self.calculate_difference(self.modulated, self.reference)
    

    def set_x_axis(self, 
            initial_value: float,
            loop_counter_start: int, 
            loop_counter_increment: int, 
            length_per_counter: float, 
            n_points: int, 
            num_continuous: int,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.x_initial_value = initial_value
        self.loop_counter_start = loop_counter_start
        self.loop_counter_increment = loop_counter_increment
        self.length_per_counter = length_per_counter
        self.n_points = n_points
        self.num_continuous = num_continuous

        self.x_discrete, self.x_continuous, self.loop_counters = _set_x_axis(initial_value, 
                                                            loop_counter_start, 
                                                            loop_counter_increment, 
                                                            length_per_counter, 
                                                            n_points, 
                                                            num_continuous)
        
        return self.x_discrete, self.x_continuous, self.loop_counters


    def calculate_difference(self, 
                             modulated: np.ndarray,
                             reference: np.ndarray, 
                             ) -> np.ndarray:     
        self.modulated = modulated
        self.reference = reference 
        self.difference = _calculate_difference(modulated, reference)

        return self.difference

    
    def to_fid(self, 
               filename: str, 
               key: str = 'data',
               ) -> None:
      
        _to_fid(filename, getattr(self, key))
    

    def to_txt(self, 
               filename: str, 
               key: str = 'data',
               order: Literal['C', 'F'] = 'C',
               fmt: str ='%.6f', 
               ) -> None:

        _to_txt(filename, getattr(self, key), order=order, fmt=fmt)


@dataclass
class Fid_triple:
    """
    Template calss for 1D intensity data with triple-FID acquisition, such as compensated REDOR. 
    """
    filename: str
    alpha: float = 1

    data: np.ndarray = field(init=False)
    modulated: np.ndarray = field(init=False)
    compensated: np.ndarray = field(init=False)
    difference: np.ndarray = field(init=False)
    reference: np.ndarray = field(init=False)

    x_initial_value: float = 0
    loop_counter_start: int | None = None
    loop_counter_increment: int | None = None
    length_per_counter: float | None = None
    n_points: int | None = None
    num_continuous: int = 100

    x_discrete: np.ndarray = field(init=False)
    x_continuous: np.ndarray = field(init=False)
    loop_counters: np.ndarray = field(init=False)

    load_text_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialization: read data from .txt or .fid file. 

        Parameters:
        - filename: str, input file path
        - **kwargs: additional keyword arguments for np.loadtxt() when reading .txt file, see https://numpy.org/doc/2.1/reference/generated/numpy.loadtxt.html
        """
        if PurePath(self.filename).suffix == '.txt':
            self.data = np.loadtxt(self.filename, **self.load_text_options)
        elif PurePath(self.filename).suffix == '.fid':
            self.data, _ = read_fid(self.filename)
        else:
            raise TypeError("File must be .txt or .fid! ")
        
        if self.data.shape[0] % 3 == 1:
            raise ValueError("Data length is not divisible by 3, cannot be reshaped into triples.")
        self.data = np.reshape(self.data, (3, -1))
        self.modulated = self.data[0]
        self.compensated = self.data[1]
        self.reference = self.data[2]

        self.calculate_difference(self.modulated, self.compensated, self.reference)


    def set_x_axis(self, 
            initial_value: float,
            loop_counter_start: int, 
            loop_counter_increment: int, 
            length_per_counter: float, 
            n_points: int, 
            num_continuous: int,
            ) -> None:
        
        self.x_initial_value = initial_value
        self.loop_counter_start = loop_counter_start
        self.loop_counter_increment = loop_counter_increment
        self.length_per_counter = length_per_counter
        self.n_points = n_points
        self.num_continuous = num_continuous

        self.x_discrete, self.x_continuous, self.loop_counters = _set_x_axis(initial_value, 
                                                            loop_counter_start, 
                                                            loop_counter_increment, 
                                                            length_per_counter, 
                                                            n_points, 
                                                            num_continuous)


    def calculate_difference(self, 
                             modulated: np.ndarray,
                             compensated: np.ndarray,
                             reference: np.ndarray, 
                             ) -> None:     
        
        self.modulated = modulated
        self.compensated = compensated
        self.reference = reference 

        self.difference_not_compensated = _calculate_difference(modulated, reference)
        self.difference = self.difference_not_compensated + self.alpha * _calculate_difference(compensated, reference)

    
    def to_fid(self, 
               filename: str, 
               key: str = 'data',
               ) -> None:

        _to_fid(filename, getattr(self, key))
    

    def to_txt(self, 
               filename: str, 
               key: str = 'data',
               order: Literal['C', 'F'] = 'C',
               fmt: str ='%.6f', 
               ) -> None:

        _to_txt(filename, getattr(self, key), order=order, fmt=fmt)


@dataclass
class CTDrenar(Fid_single):
    l0: int | None = None
    spin_rate: float | None = None

    phase_range: tuple[float, float] | None = None
    phase_increment: float | None = None
    phase_continuous: np.ndarray = field(init=False)
    phase_discrete: np.ndarray = field(init=False)

    dephasing: np.ndarray = field(init=False)
    dephasing_time: float | None = None
    verbose: bool = False
    """
    A class for handling CT-DRENAR data, which is a single-FID acquisition experiment with phase incrementation.

    Attributes:
    - filename: str, input file path
    - phase_range: tuple, (min_phase, max_phase) in degrees
    - phase_increment: float, increment of phase in degrees 
    - n_points: int, number of points in the experiment
    - idx_reference: int, index of the reference point in the data (default: middle point if n_points is odd, otherwise not defined)
    - l0: int, rotor cycles for the first point (default: None, must be provided for calculating dephasing time)
    - spin_rate: float, spinning rate in kHz (default: None, must be provided for calculating dephasing time)
    - verbose: bool, whether to print detailed information during initialization (default: False)

    - data: np.ndarray, raw data read from the file
    - phase_discrete: np.ndarray, discrete phase angles corresponding to the data points
    - phase_continuous: np.ndarray, continuous phase angles for plotting the fit
    - dephasing: np.ndarray, dephasing data (S)
    - reference: np.ndarray, reference data (S₀)
    - difference: np.ndarray, normalized difference (1 - S/S₀)

    - dephasing_time: float, dephasing time calculated from l0 and spin rate (if provided)
    """

    def __post_init__(self) -> None:
        """
        Initialization: read data from .txt or .fid file and process the data to generate difference (normalized DQ intensity) array and the phase angles for each point.
        
        Parameters:
        - filename: str, input file path
        - phase_range: tuple, (min_phase, max_phase) in degrees (default: None, will be set to (0, 180) if phase_increment is not provided)
        - phase_increment: float, increment of phase in degrees (default: None, will be calculated from phase_range and n_points if not provided)
        - n_points: int, number of points in the experiment (default: None, will be set to the length of data if not provided)
        - idx_reference: int, index of the reference point in the data (default: None, will be set to the middle point if n_points is odd, otherwise not defined)
        - l0: int, rotor cycles for the first point (default: None, must be provided for calculating dephasing time)
        - spin_rate: float, spinning rate in kHz (default: None, must be provided for calculating dephasing time)
        - verbose: bool, whether to print detailed information during initialization (default: False)
        - **kwargs: additional keyword arguments for np.loadtxt() when reading .txt file, see https://numpy.org/doc/2.1/reference/generated/numpy.loadtxt.html
        """
        super().__post_init__()

        if self.n_points is not None and self.n_points != np.shape(self.data)[0]:
            raise ValueError("Inconsistent number of points between input and data.")
        else:
            self.n_points = np.shape(self.data)[0]

        if self.reference_idx is None:
            if self.n_points % 2 == 0:
                warnings.warn("Even number of points, the reference point cannot be found.")
            else:
                self.reference_idx = self.n_points // 2   # python index, starting from 0
                self.calculate_difference(self.reference_idx)

        self.phase_range, self.phase_increment = _set_phases(self.phase_range, self.phase_increment, self.n_points)
        self.phase_discrete = np.linspace(self.phase_range[0], self.phase_range[1], num=self.n_points)
        self.phase_continuous = np.linspace(self.phase_range[0], self.phase_range[1], num=100)

        self.phase_discrete, self.phase_continuous = self.set_x_axis(self.phase_range[0], 
                                                                     0, 
                                                                     1, 
                                                                     self.phase_increment, 
                                                                     self.n_points, 
                                                                     self.num_continuous)

        if isinstance(self.l0, Number) and isinstance(self.spin_rate, Number): 
            self.dephasing_time = 16 * self.l0 / self.spin_rate


    def plot_difference(self, 
                        xlim: tuple[float, float] | None = None, 
                        ylim: tuple[float, float] | None = None, 
                        figure_size: tuple[float, float] = (4, 4),
                        show_legend: bool = False, 
                        **kwargs: Any
                        ) -> tuple[Figure, Axes]:
        """
        Plot the difference (1 - S/S₀) against the phase angle.

        Parameters:
        - xlim: tuple, limits for x-axis
        - ylim: tuple, limits for y-axis
        - figure_size: tuple, size of the figure (width, height)
        - show_legend: bool, whether to show legend
        - **kwargs: additional keyword arguments for plt.plot()

        Returns: 
        - fig: Figure, the created figure object
        - ax: Axes, the created axes object
        """
        if not hasattr(self, 'phase_discrete') or not hasattr(self, 'difference'):
            raise AttributeError("Data does not exist. Read result file again with CTDrenar(filename).")

        fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
        ax.plot(self.phase_discrete, self.difference, **kwargs)
        ax.set_xlabel('Phase shift (°)')
        ax.set_ylabel('Normalized DQ intensity')
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if show_legend:
            ax.legend()
        
        plt.show()

        return fig, ax

                
    def fit(self, 
            function: Callable = ctdrenar, 
            ) -> float:
        """
        Fit CT-DRENAR data with the analytical function and calculate the effective dipolar coupling constant.

        Parameters:
        - function: callable, the function to fit the data (default: ctdrenar)

        Returns:
        - beff_opt: float, effective dipolar coupling constant calculated from the optimized z-value and the dephasing time (if provided), in kHz
        - z_opt: float, optimized z-value from the fitting, in ms^2, if dephasing time is not provided
        """
        if not hasattr(self, 'dephasing_time'):
            if isinstance(self.l0, Number) and isinstance(self.spin_rate, Number):
                self.dephasing_time = 16 * self.l0 / self.spin_rate
            else:
                warnings.warn("l0 and spin_rate must be provided to calculate dephasing time for fitting. Do so when initializing the class or add them as attributes seperately.")
        
        self.popt, self.pconv = curve_fit(function, self.phase_discrete, self.difference, bounds=(0, np.inf))
        #perr = np.sqrt(np.diag(pconv))  # standard deviation
        self.z_opt = self.popt[0]
        self.predict = function(self.phase_continuous, self.z_opt)

        if self.dephasing_time is not None:
            self.beff_opt = np.sqrt(self.z_opt)/self.dephasing_time
            print(f'Effective dipolar coupling constant by analytical fitting = {self.beff_opt:.3f} kHz')
            #r_opt = (mu_0 / (4*pi) * (gamma_I*gamma_S*hbar) / (2*pi) / d_opt /1000)**(1/3) * 10**9  # in nm
            #print(f'Effective distance r = {r_opt:.3f} nm')
            return self.beff_opt
        else:
            warnings.warn("Dephasing time is not provided, returning only the optimized z-value.")
            return self.z_opt


    def plot_fit(self, 
                xlim: tuple[float, float] | None = None, 
                ylim: tuple[float, float] | None = None, 
                figure_size: tuple[float, float] = (4, 4),
                color = '#0092c8', 
                show_legend: bool = False, 
                ) -> tuple[Figure, Axes]:
        """
        Plotting the fitted curve together with the experimental data.

        Parameters:
        - xlim: tuple, limits for x-axis
        - ylim: tuple, limits for y-axis
        - figure_size: tuple, size of the figure (width, height)
        - color: str, color for both experimental data and fitted curve (default: '#0092c8')
        - show_legend: bool, whether to show legend
        
        Returns:
        - fig: Figure, the created figure object
        - ax: Axes, the created axes object
        """
        if not hasattr(self, 'phase_continuous') or not hasattr(self, 'predict'):
            raise AttributeError("Fitted data does not exist, do the fitting first.")

        fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
        ax.plot(self.phase_discrete, self.difference, marker='o', linestyle='none', color=color, label='experiment')
        ax.plot(self.phase_continuous, self.predict, marker='none', linestyle='-', color=color, label='analytical fit')
        ax.set_xlabel('Phase shift (°)')
        ax.set_ylabel('Normalized DQ intensity')
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if show_legend:
            ax.legend()

        plt.show()
        
        return fig, ax


@dataclass
class Redor(Fid_pair):
    """
    A class for handling REDOR data, which is a double-FID acquisition experiment with time incrementation. 
    """
    l0: int | None = None
    l10: int | None = None
    spin_rate: float | None = None

    gamma_I: float | None = None
    gamma_S: float | None = None

    truncated_n_points: int | None = None
    n_rotor_cycles: np.ndarray = field(init=False)
    time_discrete: np.ndarray = field(init=False)
    time_continuous: np.ndarray = field(init=False)

    dephasing: np.ndarray = field(init=False)
    reference: np.ndarray = field(init=False)
    difference: np.ndarray = field(init=False)

    popt: np.ndarray = field(init=False)
    pconv: np.ndarray = field(init=False)
    z_opt: float = field(init=False)
    d_opt: float = field(init=False)
    r_opt: float = field(init=False)

    verbose: bool = False
      

    def __post_init__(self):
        """
        Initialization: read data from .txt or .fid file and process the data to generate difference array and the time axis for each point. 
        The time axis is calculated based on l0, l10, and spin rate if they are provided.

        Parameters:
        - filename: str, input file path
        - l0: int, rotor cycles for the first point (default: None, must be provided for calculating dephasing time)
        - l10: int, increment constant defined in the pulse program (default: None, must be provided for calculating dephasing time)
        - spin_rate: float, spinning rate in kHz (default: None, must be provided for calculating dephasing time)
        - gamma_I: float, gyromagnetic ratio of the observed nucleus in MHz/T (default: None, must be provided for calculating distance from coupling constant)
        - gamma_S: float, gyromagnetic ratio of the dephasing nucleus in MHz/T (default: None, must be provided for calculating distance from coupling constant)
        - verbose: bool, whether to print detailed information during initialization (default: False)
        """

        if PurePath(self.filename).suffix == '.txt':
            self.data = np.loadtxt(self.filename)
            if self.data.shape[0] % 2 == 1:
                raise ValueError("Data length is odd, cannot be reshaped into pairs.")
            self.data = np.reshape(self.data, (2, -1))

            if self.truncated_n_points is None:
                self.n_points = np.shape(self.data)[1]
            else:
                self.n_points = self.truncated_n_points

            self.dephasing = self.data[0,0:self.n_points]
            self.reference = self.data[1,0:self.n_points]
            
        elif PurePath(self.filename).suffix == '.fid':
            self.data, _ = read_fid(self.filename)
            if self.truncated_n_points is None:
                self.n_points = np.shape(self.data)[0] - 1   
            else:
                self.n_points = self.truncated_n_points     
            
            self.dephasing = self.data[1:self.n_points+1]
            self.reference = np.ones(self.n_points) * self.data[0]

        else:
            raise TypeError("File must be .txt or .fid! ")

        # deprecated: switch defination of attributes
        self.truncated_n_points = self.n_points

        if np.any(self.reference == 0):
            raise ValueError("Reference contains zero(s), cannot divide.")
        self.difference = 1 - self.dephasing/self.reference

        print("\nList of data: ") if self.verbose else ""
        print(self.data) if self.verbose else ""
        print("\nList of differences: ") if self.verbose else ""
        print(self.difference) if self.verbose else ""
        
        if isinstance(self.l0, Number) and isinstance(self.l10, Number) and isinstance(self.spin_rate, Number):
            self.time_discrete, self.time_continuous, self.loop_counters = self.set_x_axis(0, 
                            self.l0 + 1, 
                            self.l10 * 2, 
                            1/self.spin_rate, 
                            self.n_points, 
                            self.num_continuous)
            print(f"\n{self.n_points :d} steps, time increment {2*self.l10} rotor cycles, {self.time_discrete[1] - self.time_discrete[0]:.3f} ms for each step.") if self.verbose else ""
            # For different pulse sequences, the time axis can also be defined by calling the .set_x_axis() method using more general loop counter settings. 
    

    def plot_difference(self, 
                        xlim: tuple[float, float] | None = None, 
                        ylim: tuple[float, float] | None = None, 
                        figure_size: tuple[float, float] = (4, 4),
                        show_legend: bool = False, 
                        **kwargs: Any
                        ) -> tuple[Figure, Axes]:
        """
        Plot the difference (1 - S/S₀) against the phase angle.

        Parameters:
        - xlim: tuple, limits for x-axis
        - ylim: tuple, limits for y-axis
        - figure_size: tuple, size of the figure (width, height)
        - show_legend: bool, whether to show legend
        - **kwargs: additional keyword arguments for plt.plot()

        Returns: 
        - fig: Figure, the created figure object
        - ax: Axes, the created axes object
        """
        if self.time_discrete is None:
            if isinstance(self.l0, Number) and isinstance(self.l10, Number) and isinstance(self.spin_rate, Number):
                self.set_x_axis(0, 
                                self.l0 + 1, 
                                self.l10 * 2, 
                                1/self.spin_rate, 
                                self.n_points, 
                                self.num_continuous)
                print(f"\n{self.n_points :d} steps, time increment {2*self.l10} rotor cycles, {self.time_discrete[1] - self.time_discrete[0]:.3f} ms for each step.") if self.verbose else ""
            else:
                raise AttributeError("Time axis does not exist. Please specify l0, l10, and n_points.")

        fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
        ax.plot(self.time_discrete, self.difference, **kwargs)
        ax.set_xlabel('Recoupling time (ms)')
        ax.set_ylabel('1 - S/S₀')
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if show_legend:
            ax.legend()
        
        plt.show()

        return fig, ax
        

    def fit(self, 
            function: Callable = redor_bessel(5), 
            ) -> tuple[float, float | None]:
        """
        Fit REDOR data with the analytical function and calculate the effective dipolar coupling constant.

        Parameters:
        - function: callable, the function to fit the data (default: ctdrenar)

        Returns:
        - beff_opt: float, effective dipolar coupling constant from fitting, in kHz
        - r_opt: float, optimized distance from the fitting, in Å, if time axis is not provided
        """
        self.popt, self.pconv = curve_fit(function, self.time_discrete, self.difference, bounds=(0, np.inf))
        #perr = np.sqrt(np.diag(pconv))  # standard deviation
        self.beff_opt = self.popt[0]
        self.predict = function(self.time_continuous, self.beff_opt)
        print(f'Effective dipolar coupling constant by analytical fitting = {self.beff_opt*1000:.1f} Hz') if self.verbose else ""

        if self.gamma_I is None or self.gamma_S is None:

            return self.beff_opt, None
        else:
            mu_0 = physical_constants['vacuum mag. permeability'][0]
            hbar = physical_constants['reduced Planck constant'][0]
            self.r_opt = (mu_0 / (4*pi) * abs(self.gamma_I*self.gamma_S*hbar) / (2*pi) / self.beff_opt /1000)**(1/3) * 10**10  # in angstrom
            print(f'Effective distance by analytical fitting = {self.r_opt:.3f} Å') if self.verbose else ""

            return self.beff_opt, self.r_opt
        

    def plot_fit(self, 
                xlim: tuple[float, float] | None = None, 
                ylim: tuple[float, float] | None = None, 
                figure_size: tuple[float, float] = (4, 4),
                color = '#0092c8', 
                show_legend: bool = False, 
                ) -> tuple[Figure, Axes]:
        """
        Plotting the fitted curve together with the experimental data.

        Parameters:
        - xlim: tuple, limits for x-axis
        - ylim: tuple, limits for y-axis
        - figure_size: tuple, size of the figure (width, height)
        - color: str, color for both experimental data and fitted curve (default: '#0092c8')
        - show_legend: bool, whether to show legend
        
        Returns:
        - fig: Figure, the created figure object
        - ax: Axes, the created axes object
        """
        fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
        ax.plot(self.time_discrete, self.difference, marker='o', linestyle='none', color=color, label='experiment')
        ax.plot(self.time_continuous, self.predict, marker='none', linestyle='-', color=color, label='analytical fit')
        ax.set_xlabel('Recoupling time (ms)')
        ax.set_ylabel('1 - S/S₀')

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if show_legend:
            ax.legend()

        plt.show()
        
        return fig, ax


@dataclass
class Redor3(Fid_triple): 
    """
    Class for handling compensated REDOR data, see https://doi.org/10.1006/jmre.2000.2191
    """
    l0: int | None = None
    l10: int | None = None
    spin_rate: float | None = None

    gamma_I: float | None = None
    gamma_S: float | None = None

    _n_points: int = field(init=False)
    truncated_n_points: int = field(init=False)
    n_rotor_cycles: np.ndarray = field(init=False)
    time_discrete: np.ndarray = field(init=False)
    time_continuous: np.ndarray = field(init=False)

    dephasing: np.ndarray = field(init=False)
    reference: np.ndarray = field(init=False)
    difference: np.ndarray = field(init=False)

    popt: np.ndarray = field(init=False)
    pconv: np.ndarray = field(init=False)
    z_opt: float = field(init=False)
    d_opt: float = field(init=False)
    r_opt: float = field(init=False)

    verbose: bool = False

    def __post_init__(self):
        """
        Initialization: read data from .txt or .fid file and process the data to generate difference array and the time axis for each point. 
        The time axis is calculated based on l0, l10, and spin rate if they are provided.

        Parameters:
        - filename: str, input file path
        - alpha: float, compensation factor defined in the pulse program (default: 1)
        - l0: int, rotor cycles for the first point (default: None, must be provided for calculating dephasing time)
        - l10: int, increment constant defined in the pulse program (default: None, must be provided for calculating dephasing time)
        - spin_rate: float, spinning rate in kHz (default: None, must be provided for calculating dephasing time)
        - gamma_I: float, gyromagnetic ratio of the observed nucleus in MHz/T (default: None, must be provided for calculating distance from coupling constant)
        - gamma_S: float, gyromagnetic ratio of the dephasing nucleus in MHz/T (default: None, must be provided for calculating distance from coupling constant)
        - verbose: bool, whether to print detailed information during initialization (default: False)
        """
        if PurePath(self.filename).suffix == '.txt':
            self.data = np.loadtxt(self.filename)
            if self.data.shape[0] % 3 == 1:
                raise ValueError("Data length is not divisible by 3, cannot be reshaped into triplets.")
            self.data = np.reshape(self.data, (3, -1))

            if self.n_points is None:
                self._n_points = np.shape(self.data)[1]
            else:
                self._n_points = self.n_points

            self.dephasing = self.data[0,0:self._n_points]
            self.compensation = self.data[1,0:self._n_points]
            self.reference = self.data[2,0:self._n_points]

        elif PurePath(self.filename).suffix == '.fid':
            self.data, _ = read_fid(self.filename)
            if self.n_points is None:
                self._n_points = np.shape(self.data)[0] - 1        
            else:
                self._n_points = self.n_points
            
            self.dephasing = self.data[1:self._n_points+1]
            self.compensation = np.ones(self._n_points) * self.data[0]
            self.reference = self.compensation

        else:
            raise TypeError("File must be .txt or .fid! ")      
            
        if np.any(self.reference == 0):
            raise ValueError("Reference contains zero(s), cannot divide.")
        
        self.difference = 1 + self.alpha - self.dephasing/self.reference - self.alpha*self.compensation/self.reference

        print("\nList of data: ") if self.verbose else ""
        print(self.data) if self.verbose else ""
        print("\nList of differences: ") if self.verbose else ""
        print(self.difference) if self.verbose else ""
        
        if isinstance(self.l0, Number) and isinstance(self.l10, Number) and isinstance(self.spin_rate, Number):
            self.set_x_axis(0, 
                            self.l0 + 1, 
                            self.l10 * 2, 
                            1/self.spin_rate, 
                            self._n_points, 
                            self.num_continuous)
            print(f"\n{self._n_points :d} steps, time increment {2*self.l10} rotor cycles, {self.time_discrete[1] - self.time_discrete[0]:.3f} ms for each step.") if self.verbose else ""
            # For different pulse sequences, the time axis can also be defined by calling the .set_x_axis() method using more general loop counter settings. 


@dataclass
class DoubleQuantum(Fid_pair): 
    l0: int | None = None
    l10: int | None = None
    spin_rate: float | None = None

    verbose: bool = False

    def __post_init__(self):
        """
        Initialization: read data from .txt or .fid file and process the data to generate difference array and the time axis for each point. 
        The time axis is calculated based on l0, l10, and spin rate if they are provided.

        Parameters:
        - filename: str, input file path
        - l0: int, rotor cycles for the first point (default: None, must be provided for calculating dephasing time)
        - l10: int, increment constant defined in the pulse program (default: None, must be provided for calculating dephasing time)
        - spin_rate: float, spinning rate in kHz (default: None, must be provided for calculating dephasing time)
        - gamma_I: float, gyromagnetic ratio of the observed nucleus in MHz/T (default: None, must be provided for calculating distance from coupling constant)
        - gamma_S: float, gyromagnetic ratio of the dephasing nucleus in MHz/T (default: None, must be provided for calculating distance from coupling constant)
        - verbose: bool, whether to print detailed information during initialization (default: False)
        """
        super().__post_init__()

        print("\nList of data: ") if self.verbose else ""
        print(self.data) if self.verbose else ""
        print("\nList of differences: ") if self.verbose else ""
        print(self.difference) if self.verbose else ""
        
        if isinstance(self.l0, Number) and isinstance(self.l10, Number) and isinstance(self.spin_rate, Number):
            self.time_discrete, self.time_continuous, self.loop_counters = self.set_x_axis(0, 
                            self.l0, 
                            self.l10, 
                            self.spin_rate, 
                            self.n_points, 
                            self.num_continuous,
                            )
            print(f"\n{self.n_points :d} steps, time increment {2*self.l10} rotor cycles, {self.time_discrete[1] - self.time_discrete[0]:.3f} ms for each step.") if self.verbose else ""