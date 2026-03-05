"""
Useful python classes and functions for handling dipolar NMR experiments. 

Author(s): Yufei Wu, Julius Schlueter
Created on: 2025/08/08
Last modified: 2026/03/05
"""

import os
from pathlib import PurePath
import warnings
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .functions import ctdrenar

# packages for type-hint
from typing import Any, Callable
from numbers import Number
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def read_fid(
        filename: str
        ) -> tuple[np.ndarray, np.ndarray]:
    
    with open(filename, 'r') as f:
        data = []
        n_rows = None
        in_data_section = False

        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('NP='):
                    n_rows = int(line.split('=')[1])
                elif line == 'DATA':
                    in_data_section = True
                elif line == 'END':
                    break
                elif in_data_section:
                    if line:  # skip empty lines
                        values = [float(x) for x in line.split()]
                        data.append(values)
                        
        if n_rows is None:
            raise ValueError("NP value not found in the .fid file.")

    data_real = np.array(data)[:, 0]  # use first column
    data_imag = np.array(data)[:, 1]  # use second column if needed

    return data_real, data_imag


def set_phases(
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


class Fid_single:
    """
    Template calss for 1D intensity data with single-FID acquisition, such as CT-DRENAR. 

    Attributes:
    - filename: str, input file path
    - data: np.ndarray, raw data read from the file
    """
    def __init__(self, 
                 filename: str, 
                 **kwargs, 
                 ) -> None:
        """
        Initialization: read data from .txt or .fid file. 

        Parameters:
        - filename: str, input file path
        - **kwargs: additional keyword arguments for np.loadtxt() when reading .txt file, see https://numpy.org/doc/2.1/reference/generated/numpy.loadtxt.html
        """
        self.filename = filename
        if PurePath(filename).suffix == '.txt':
            self.data = np.loadtxt(filename, **kwargs)
        elif PurePath(filename).suffix == '.fid':
            self.data, _ = read_fid(filename)
        else:
            raise TypeError("File must be .txt or .fid! ")


    def to_fid(self, 
               filename: str, 
               key: str = 'data',
               ) -> None:
        """
        Export data to a .fid file.

        Parameters:
        - filename: str, output file path
        - key: str, attribute name of the data to be exported (default: 'data')

        Returns: 
        - None
        """
        
        if not hasattr(self, key):
            raise AttributeError("The specified key does not exist.")
        
        with open(filename, 'w') as f:
            f.write('SIMP\n')
            f.write(f'NP={np.shape(getattr(self, key))[0]}\n')
            f.write('SW=100000\n')
            f.write('TYPE=FID\n')
            f.write('DATA\n')
            for value in getattr(self, key):
                f.write(f'{value} 0\n')
            f.write('END\n')
    

    def to_txt(self, 
               filename: str, 
               key: str = 'data',
               order: str = 'C',
               fmt: str ='%.6f', 
               ) -> None:
        """
        Export data to a .txt file.

        Parameters:
        - filename: str, output file path
        - key: str, attribute name of the data to be exported (default: 'data')
        - order: str, order of reshaping the data, 'C' for row-major, 'F' for column-major (default: 'C')
        - fmt: str, formatting for float values

        Returns:
        - None
        """
        if not hasattr(self, key):
            raise AttributeError("The specified key does not exist.")

        data = getattr(self, key).reshape(-1, order=order)
        np.savetxt(filename, data, fmt=fmt)


class Fid_pair(Fid_single):
    """
    Template calss for 1D intensity data with double-FID acquisition, such as REDOR, VT-DRENAR. 

    Attributes:
    - filename: str, input file path
    - data: np.ndarray, raw data read from the file
    """
    def __init__(self, 
                 filename: str, 
                 **kwargs, 
                 ) -> None:
        """
        Initialization: read data from .txt or .fid file. 

        Parameters:
        - filename: str, input file path
        - **kwargs: additional keyword arguments for np.loadtxt() when reading .txt file, see https://numpy.org/doc/2.1/reference/generated/numpy.loadtxt.html
        """
        super().__init__(filename, **kwargs)
        if self.data.shape[0] % 2 == 1:
            raise ValueError("Data length is odd, cannot be reshaped into pairs.")
        self.data = np.reshape(self.data, (2, -1))


class Fid_triple(Fid_single):
    """
    Template calss for 1D intensity data with triple-FID acquisition, such as compensated REDOR. 

    Attributes:
    - filename: str, input file path
    - data: np.ndarray, raw data read from the file
    """
    def __init__(self, 
                 filename: str, 
                 **kwargs, 
                 ) -> None:
        """
        Initialization: read data from .txt or .fid file. 

        Parameters:
        - filename: str, input file path
        - **kwargs: additional keyword arguments for np.loadtxt() when reading .txt file, see https://numpy.org/doc/2.1/reference/generated/numpy.loadtxt.html
        """
        super().__init__(filename, **kwargs)
        if self.data.shape[0] % 3 != 0:
            raise ValueError("Data length is not divisible by 3, cannot be reshaped into triples.")
        self.data = np.reshape(self.data, (3, -1))


class CTDrenar(Fid_single):
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

    def __init__(self, 
                 filename: str, 
                 phase_range: tuple[float, float] | None = None,
                 phase_increment: float | None = None, 
                 n_points: int | None = None,
                 idx_reference: int | None = None,
                 l0: int | None = None,
                 spin_rate: float | None = None,
                 verbose=False, 
                 **kwargs: Any, 
                 ) -> None:
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
        self.phase_range = phase_range
        self.verbose = verbose

        super().__init__(filename, **kwargs)
        self.n_points = n_points if n_points is not None else np.shape(self.data)[0]
        self.phase_range, self.phase_increment = set_phases(phase_range, phase_increment, self.n_points)

        if idx_reference is not None:
            self.idx_reference = idx_reference
        elif self.n_points %2 == 0:
                warnings.warn("Even number of points, the reference point cannot be found.")
        else:
            self.idx_reference = self.n_points // 2   # python index, starting from 0
        
        self.phase_discrete = np.linspace(self.phase_range[0], self.phase_range[1], num=self.n_points)
        self.phase_continuous = np.linspace(self.phase_range[0], self.phase_range[1], num=100)
        
        self.dephasing = self.data
        self.reference = np.ones(self.n_points) * self.data[self.idx_reference]
        if np.any(self.reference == 0):
            raise ValueError("Reference contains zero(s), cannot divide.")
        self.difference = 1 - self.dephasing/self.reference

        self.l0 = l0
        self.spin_rate = spin_rate
        if isinstance(self.l0, Number) and isinstance(self.spin_rate, Number): 
            self.dephasing_time = 16 * self.l0 / self.spin_rate
        else:
            self.dephasing_time: float | None = None


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
    def __init__(self, file_path, l10, spin_rate, l0=1, gamma=[None, None], n_points=None, verbose=False, ):
        
        self.file_path = file_path
        self.l10 = l10
        self.l0 = l0
        self.spin_rate = spin_rate
        self.gamma = gamma
        self.n_points = n_points
        self.verbose = verbose
        
        self.dephasing_time = None
        self.data = None
        self.dephasing = None
        self.reference = None
        self.difference = None
        

        self.fitting = None
        
        self.popt = None
        self.pconv = None
        self.z_opt = None
        self.d_opt = None
        self.r_opt = None
        
        try: 
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.txt':
                self.read_txt()
            elif ext == '.fid':
                self.read_fid()
            else:
                raise TypeError("File must be .txt or .fid! ")
        except TypeError as te:
            print(te)
        #except Exception as e:
            #print(f"Error reading file: {e}")
            

    def read_txt(self):
        print("reading txt") if self.verbose else ""
        with open(self.file_path, 'r') as f:
            self.data = np.loadtxt(f)
            n_rows = self.data.shape[0]
        
        if self.n_points is None:
            self.n_points = n_rows//2
        self.data = np.reshape(self.data, (2, -1))
        self.data = self.data[:,0:self.n_points]
            
        self.dephasing = self.data[0,:]
        self.reference = self.data[1,:]
        
        if np.any(self.reference == 0):
            raise ValueError("Reference contains zero(s), cannot divide.")
        self.difference = 1 - self.dephasing/self.reference
        
        self.time_discrete = 1/self.spin_rate * np.linspace(self.l0+1, 2*self.l10*(self.n_points-1)+self.l0+1, num=self.n_points)  # in ms if spin_rate in kHz
        self.time_continuous = np.linspace(0, self.time_discrete[-1], num=100)
        
        print(f"\n{self.n_points :d} steps, time increment {2*self.l10} rotor cycles, {self.time_discrete[1] - self.time_discrete[0]:.3f} ms for each step.") if self.verbose else ""

        print("\nList of data: ") if self.verbose else ""
        print(self.data) if self.verbose else ""
        print("\nList of differences: ") if self.verbose else ""
        print(self.difference) if self.verbose else ""


    def read_fid(self):
        print("reading fid") if self.verbose else ""
        data = []
        n_rows = None
        in_data_section = False

        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('NP='):
                    n_rows = int(line.split('=')[1])
                elif line == 'DATA':
                    in_data_section = True
                elif line == 'END':
                    break
                elif in_data_section:
                    if line:  # skip empty lines
                        values = [float(x) for x in line.split()]
                        data.append(values)
                        
        if n_rows is None:
            raise ValueError("NP value not found in the .fid file.")
        
        self.data = np.array(data)[:, 0]  # use first column
        
        if n_rows%2 == 1:
            warnings.warn("\nPairwise acquisition not detected. ")
        self.pair_acquisition = True
        self.n_points = n_rows - 1

        self.reference = np.ones(self.n_points) * self.data[0]
        self.dephasing = self.data[1:]

        if np.any(self.reference == 0):
            raise ValueError("Reference contains zero(s), cannot divide.")
            
        self.difference = 1 - self.dephasing/self.reference
        self.time_discrete = 1/self.spin_rate * np.linspace(self.l0+1, 2*self.l10*(self.n_points-1)+self.l0+1, num=self.n_points)  # in ms if spin_rate in kHz
        self.time_continuous = np.linspace(0, self.time_discrete[-1], num=100)
            
        print(f"\n{self.n_points :d} steps, time increment {2*self.l10} rotor cycles, {self.time_discrete[1] - self.time_discrete[0]:.3f} ms for each step.") if self.verbose else ""

        print("\nList of data: ") if self.verbose else ""
        print(self.data) if self.verbose else ""
        print("\nList of differences: ") if self.verbose else ""
        print(self.difference) if self.verbose else ""
        
        
    def to_fid(self, filename):
        if not hasattr(self, 'dephasing'):
            raise AttributeError("Data does not exist.")
        
        with open(filename, 'w') as f:
            f.write('SIMP\n')
            f.write(f'NP={self.n_points}\n')
            f.write('SW=100000\n')
            f.write('TYPE=FID\n')
            f.write('DATA\n')
            for value in self.dephasing/self.reference:
                f.write(f'{value} 0\n')
            f.write('END\n')
            
        
    def plot_difference(self, xlim=None, ylim=None, color='blue', label=None, show_legend=False, **kwargs):
        if not hasattr(self, 'time_discrete') or not hasattr(self, 'difference'):
            raise AttributeError("Data does not exist. Run read_txt() first.")

        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.scatter(self.time_discrete, self.difference)
        ax.set_xlabel('REDOR time (ms)')
        ax.set_ylabel('1 - S/S₀')
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if show_legend and label:
            ax.legend()
        
        plt.show()
        
                
    def fit(self, function):
        try: 
            if self.gamma[0] is None or self.gamma[1] is None:
                raise ValueError("Gamma values are not provided! ")
            else:
                self.gamma[0] = abs(self.gamma[0])
                self.gamma[1] = abs(self.gamma[1])
        except TypeError as te:
            print(te)
        
        from scipy.optimize import curve_fit
        from scipy.constants import pi
        from scipy.constants import physical_constants
        
        self.popt, self.pconv = curve_fit(function, self.time_discrete, self.difference, bounds=(0, np.inf))
        #perr = np.sqrt(np.diag(pconv))  # standard deviation
        self.d_opt = self.popt[0]

        print(f'Effective dipolar coupling constant by fitting = {self.d_opt:.3f} kHz')

        mu_0 = physical_constants['vacuum mag. permeability'][0]
        hbar = physical_constants['reduced Planck constant'][0]
        self.r_opt = (mu_0 / (4*pi) * (self.gamma[0]*self.gamma[1]*hbar) / (2*pi) / self.popt[0] /1000)**(1/3) * 10**9  # in nm
        print('Fitted r = {:.3f} nm'.format(self.r_opt))

        if self.dephasing_time is not None:
            self.d_opt = np.sqrt(self.z_opt)/self.dephasing_time
            print(f'Effective dipolar coupling constant by fitting = {self.d_opt:.3f} kHz')
        
        self.predict = function(self.time_continuous, self.d_opt)

        
    def plot_fit(self, xlim=None, ylim=None, color='blue', label=None, show_legend=False, **kwargs):
        if not hasattr(self, 'time_discrete') or not hasattr(self, 'difference'):
            raise AttributeError("Data does not exist. Run read_txt() first.")

        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.scatter(self.time_discrete, self.difference)
        ax.plot(self.time_continuous, self.predict)
        ax.set_xlabel('REDOR time (ms)')
        ax.set_ylabel('1 - S/S₀')
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if show_legend and label:
            ax.legend()
        
        plt.show()