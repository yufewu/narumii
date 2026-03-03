"""
Useful python classes and functions for handling DRENAR data. 

Author(s): Yufei Wu, Julius Schlueter
Created on: 2025/08/08
Last modified: 2026/01/21
"""

import os
from pathlib import PurePath
import warnings
import numpy as np
import matplotlib.pyplot as plt
import warnings


def read_fid(
        filename: str, 
        only_real: bool = True
        ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    
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

    if only_real:
        return data_real
    else:
        return data_real, data_imag


class Data_single_fid:
    def __init__(self, 
                 filename: str, 
                 only_real: bool = True,
                 **kwargs
                 ) -> None:
        
        self.filename = filename
        if PurePath(filename).suffix == '.txt':
            self.data = np.loadtxt(filename, **kwargs)
        elif PurePath(filename).suffix == '.fid':
            self.data = read_fid(filename, only_real=only_real)
        else:
            raise TypeError("File must be .txt or .fid! ")


class Data_pair_fid:
    pass


class Data_triple_fid:
    pass

class CTDrenar:
    def __init__(self, file_path, l0=None, spin_rate=None, gamma=None, verbose=False, ):
        
        self.file_path = file_path
        self.verbose = verbose
        self.l0 = l0
        self.spin_rate = spin_rate
        self.gamma = gamma
        
        self.dephasing_time = None
        self.data = None
        self.dephasing = None
        self.reference = None
        self.difference = None
        
        self.n_points = None
        self.pair_acquisition = None
        self.fitting = None
        
        self.phase_discrete = None
        self.phase_continuous = None
        self.popt = None
        self.pconv = None
        self.z_opt = None
        self.d_opt = None
        self.r_opt = None
        
        try: 
            ext = os.path.splitext(file_path)[1].lower()
            if file_path.split('.')[-1] == 'txt':
                self.read_txt()
            elif file_path.split('.')[-1] == 'fid':
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
        
        if n_rows%2 == 1:
            print("\nPairwise acquisition not detected. ") if self.verbose else ""
            self.pair_acquisition = False
            self.n_points = n_rows
            
            self.dephasing = self.data
            print(self.dephasing) if self.verbose else ""
            self.reference = np.ones(self.n_points) * self.data[(n_rows-1)//2]
        else:
            print("\nPairwise acquisition detected. ") if self.verbose else ""
            self.pair_acquisition = True
            self.n_points = n_rows//2
            self.data = np.reshape(self.data, (2, self.n_points))
            
            self.dephasing = self.data[0,:]
            self.reference = self.data[1,:]
        
        if np.any(self.reference == 0):
            raise ValueError("Reference contains zero(s), cannot divide.")
        self.difference = 1 - self.dephasing/self.reference
        self.phase_discrete = np.linspace(0, 180, num=self.n_points)
        self.phase_continuous = np.linspace(0, 180, num=100)
            
        print(f"\n{self.n_points :d} steps, phase increment {self.phase_discrete[1] - self.phase_discrete[0]:.1f} degree for each step.") if self.verbose else ""

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
            print("\nPairwise acquisition not detected. ") if self.verbose else ""
            self.pair_acquisition = False
            self.n_points = n_rows
            
            self.dephasing = self.data
            self.reference = np.ones(self.n_points) * self.data[(n_rows-1)//2]
        else:
            print("\nPairwise acquisition detected. ") if self.verbose else ""
            self.pair_acquisition = True
            self.n_points = n_rows//2
            self.data = np.reshape(self.data, (2, self.n_points))
            
            self.dephasing = self.data[0,:]
            self.reference = self.data[1,:]

        if np.any(self.reference == 0):
            raise ValueError("Reference contains zero(s), cannot divide.")
            
        self.difference = 1 - self.dephasing/self.reference
        self.phase_discrete = np.linspace(0, 180, num=self.n_points)
        self.phase_continuous = np.linspace(0, 180, num=100)
            
        print(f"\n{self.n_points :d} steps, phase increment {self.phase_discrete[1] - self.phase_discrete[0]:.1f} degree for each step.") if self.verbose else ""

        print("\nList of data: ") if self.verbose else ""
        print(self.data) if self.verbose else ""
        print("\nList of differences: ") if self.verbose else ""
        print(self.difference) if self.verbose else ""
        
    def plot_difference(self, xlim=None, ylim=None, color='blue', label=None, show_legend=False, **kwargs):
        if not hasattr(self, 'phase_discrete') or not hasattr(self, 'difference'):
            raise AttributeError("Data does not exist. Run read_txt() or read_fid() first.")

        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.scatter(self.phase_discrete, self.difference)
        ax.set_xlabel('Phase angle (degree)')
        ax.set_ylabel('1 - S/S₀')
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if show_legend and label:
            ax.legend()
        
        plt.show()
        
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
    
    def to_txt(self, filename, fmt='%.6f'):
        """
        Export data to a .txt file.

        Parameters:
        - filename: str, output file path
        - include_reference: bool, whether to write two columns (dephasing + reference)
        - header: bool, whether to write a header line
        - fmt: str, formatting for float values
        """
        if not hasattr(self, 'dephasing'):
            raise AttributeError("Data does not exist.")

        with open(filename, 'w') as f:
            for i in range(self.n_points):
                f.write(f"{fmt % self.dephasing[i]}\n")
                
    def fit(self, function):
        from scipy.optimize import curve_fit
        self.popt, self.pconv = curve_fit(function, self.phase_discrete, self.difference, bounds=(0, np.inf))
        #perr = np.sqrt(np.diag(pconv))  # standard deviation
        self.z_opt = self.popt[0]

        if self.l0 is not None and self.spin_rate is not None:
            self.dephasing_time = 16 * self.l0 / self.spin_rate

        if self.dephasing_time is not None:
            self.d_opt = np.sqrt(self.z_opt)/self.dephasing_time
            print(f'Effective dipolar coupling constant by fitting = {self.d_opt:.3f} kHz')
        
        self.predict = function(self.phase_continuous, self.z_opt)
        #r_opt = (mu_0 / (4*pi) * (gamma_I*gamma_S*hbar) / (2*pi) / d_opt /1000)**(1/3) * 10**9  # in nm
        #print(f'Effective distance r = {r_opt:.3f} nm')
        
    def plot_fit(self, xlim=None, ylim=None, color='blue', label=None, show_legend=False, **kwargs):
        if not hasattr(self, 'phase_discrete') or not hasattr(self, 'difference'):
            raise AttributeError("Data does not exist. Run read_txt() first.")

        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        ax.scatter(self.phase_discrete, self.difference)
        ax.plot(self.phase_continuous, self.predict)
        ax.set_xlabel('Phase angle (degree)')
        ax.set_ylabel('1 - S/S₀')
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if show_legend and label:
            ax.legend()
        
        plt.show()


class Redor:
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