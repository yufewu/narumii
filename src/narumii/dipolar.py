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
    Private function for calculating the phase-related parameters with user-input phase ranges or increments. 
    Used by other classes. Probably only for CT-DRENAR. 

    Parameters
    ----------
    phase_range: tuple[float, float], optional
        (min_phase, max_phase) in degrees
    phase_increment: float, optional
        increment of phase in degrees
    n_points: int
        number of points in the experiment

    Returns
    -------
    phase_range: tuple[float, float]
        (min_phase, max_phase) in degrees
    phase_increment: float
        increment of phase in degrees
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
    Private function for calculating the x-axis values with user-input loop counters. Used by other classes. 

    Parameters
    ----------
    initial_value: float
        Initial value for x-axis.
    loop_counter_start: int
        Starting value of loop counter.
    loop_counter_increment: int
        Increment of loop counter.
    length_per_counter: float
        Length per loop counter.
    n_points: int
        Number of points in the experiment.
    num_continuous: int, optional
        How many points is generated for an artificial x-axis used for trendlines. Default is 100. 

    Returns
    -------
    x_discrete: np.ndarray
        Discrete x-axis values for plotting experimental/simulated data.
    x_continuous: np.ndarray
        Continuous x-axis values for plotting lines (predicted data). 
    loop_counters: np.ndarray
        Loop counter array in case needed.
    """
    loop_counters = np.linspace(loop_counter_start, 
                                 loop_counter_increment * (n_points - 1) + loop_counter_start, 
                                 num=n_points)
    x_discrete = loop_counters * length_per_counter + initial_value
    x_continuous = np.linspace(x_discrete[0], x_discrete[-1], num=num_continuous)

    return x_discrete, x_continuous, loop_counters


def _calculate_difference(modulated: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Private function for calculating the normalized difference (1 - S'/S₀) between modulated and reference signals. Used by other classes.

    Parameters
    ----------
    modulated: np.ndarray
        Modulated signal (S'), 1D array.
    reference: np.ndarray
        Reference signal (S₀), 1D array.

    Returns
    -------
    difference: np.ndarray
        Normalized difference (1 - S'/S₀), 1D array.
        
    """
    if np.any(reference == 0):
        raise ValueError("Reference contains zero(s), cannot divide.")
    return 1 - modulated / reference


def _to_fid(filename: str, 
           data: np.ndarray, 
           ) -> None:
    """
    Private function for exporting data to a .fid file.

    Parameters
    ----------
    filename: str
        output file path
    data: np.ndarray
        data to be exported

    Returns
    -------
    None
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
            fmt: str | list[str] ='%.6f', 
            ) -> None:
    """
    Private function for exporting data to a .txt file.

    Parameters
    ----------
    filename: str
        output file path
    data: np.ndarray
        data to be exported
    order: str, optional
        order of reshaping the data, 'C' for row-major, 'F' for column-major (default: 'C')
    fmt: str or sequence of strs, optional
        Extra parameter for np.savetxt, controls the format of the output. Check https://numpy.org/devdocs/reference/generated/numpy.savetxt.html \
            for more. Default is '%.6f'. 

    Returns
    -------
    None
    """
    data = data.reshape((-1, ), order=order)
    np.savetxt(filename, data, fmt=fmt)

@dataclass
class Fid_single:
    """
    Template class for 1D intensity data with single-FID acquisition, such as CT-DRENAR.

    Parameters
    ----------
    filename: str
        Input data file.
    reference_idx: int, optional
        Index representing the position of the reference point in the data array. Default None.
    load_text_options: dict, optional
        Additional keyword arguments for np.loadtxt(). Default {}.

    Attributes
    ----------
    filename: str
        See Parameters. 
    reference_idx: int
        See Parameters. 
    load_text_options: dict
        See Parameters. 

    data: np.ndarray
        Raw data read from the file, 1D array.
    n_points: int
        Number of points in the experiment.
    modulated: np.ndarray
        Modulated signal (S'), same as data for single-FID acquisition, 1D array.
    reference: np.ndarray
        Reference signal (S₀), 1D array.
    difference: np.ndarray
        Normalized difference (1 - S'/S₀), 1D array.

    x_initial_value: float
        Initial value for x-axis. Default 0. 
    loop_counter_start: int
        Starting value of loop counter. Default None.
    loop_counter_increment: int
        Increment of loop counter. Default None.
    length_per_counter: float
        Length of the real x-axis represented by each loop counter. Default None.
    num_continuous: int
        How many points is generated for an artificial x-axis used for trendlines. Default is 100. 

    x_discrete: np.ndarray
        Discrete x-axis values corresponding to the data points, 1D array.
    x_continuous: np.ndarray
        Continuous x-axis values for plotting the fitted curve, 1D array.
    loop_counters: np.ndarray
        Loop counter values, 1D array.

    Methods
    -------
    set_x_axis
        Create the x-axis corresponding to the data array. 
    calculate_difference
        Calculate the relative difference between the data array and a reference. 
    to_fid
        Export the data array to an fid file. 
    to_txt
        Export the data array to a txt file. 

    Examples
    --------
    >>> fid = Fid_single("sample.fid")
    >>> fid.calculate_difference(reference_idx=5)
    >>> print(fid.difference)
    >>> fid.to_txt("output.txt")
    """
    filename: str
    reference_idx: int | None = None

    data: np.ndarray = field(init=False)
    n_points: int = field(init=False)
    modulated: np.ndarray = field(init=False)
    reference: np.ndarray = field(init=False)
    difference: np.ndarray = field(init=False)

    x_initial_value: float = field(init=False, default=0)
    loop_counter_start: int = field(init=False)
    loop_counter_increment: int = field(init=False)
    length_per_counter: float = field(init=False)
    num_continuous: int = field(init=False, default=100)

    x_discrete: np.ndarray = field(init=False)
    x_continuous: np.ndarray = field(init=False)

    load_text_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:

        if PurePath(self.filename).suffix == '.txt':
            self.data = np.loadtxt(self.filename, **self.load_text_options)
        elif PurePath(self.filename).suffix == '.fid':
            self.data, _ = read_fid(self.filename)
        else:
            raise TypeError("File must be .txt or .fid! ")
        
        self.n_points = len(self.data)
        self.modulated = self.data
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
                num_continuous: int = 100,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create the x-axis corresponding to the data array. In the form of x_initial_value + length_per_counter*[list of \
            loop counter with defined start value, increment, and number of points]. 

        Parameters
        ----------
        x_initial_value: float
            Initial value for x-axis.
        loop_counter_start: int
            Starting value of loop counter.
        loop_counter_increment: int
            Increment of loop counter.
        length_per_counter: float
            Length per loop counter.
        n_points: int
            Number of points in the experiment.
        num_continuous: int, optional
            How many points is generated for an artificial x-axis used for trendlines. Default is 100. 

        Returns
        -------
        self.x_discrete: np.ndarray
            Discrete x-axis values for plotting experimental/simulated data.
        self.x_continuous: np.ndarray
            Artificial x-axis with (usually) more points used for trendlines. 
        self.loop_counters: np.ndarray
            Loop counter array in case needed. 

        Examples
        --------
        >>> fid.set_x_axis(x_initial_value=0, loop_counter_start=1, loop_counter_increment=1, length_per_counter=58.82, n_points=16)
        """
        self.x_initial_value = x_initial_value
        self.loop_counter_start = loop_counter_start
        self.loop_counter_increment = loop_counter_increment
        self.length_per_counter = length_per_counter
        self.n_points = n_points
        self.num_continuous = num_continuous

        self.x_discrete, self.x_continuous, self.loop_counters = _set_x_axis(x_initial_value, 
                                                            loop_counter_start, 
                                                            loop_counter_increment, 
                                                            length_per_counter, 
                                                            n_points, 
                                                            num_continuous)
        return self.x_discrete, self.x_continuous, self.loop_counters


    def calculate_difference(self, reference_idx: int) -> np.ndarray:
        """
        Calculate the reference and difference array from the data. 

        Parameters
        ----------
        reference_idx: int
            Index representing the position of the reference point in the data array. 

        Returns
        -------
        self.difference: np.ndarray
            Array of the relative differences between data and reference. 

        Examples
        --------
        >>> fid.calculate_difference(reference_idx=5)
        """
        self.reference_idx = reference_idx

        if self.reference_idx < 0 or self.reference_idx >= len(self.data):
            raise ValueError("Reference index is out of bounds.")
        
        self.reference = np.ones(len(self.data)) * self.data[self.reference_idx]
        self.difference = _calculate_difference(self.data, self.reference)

        return self.difference


    def to_fid(self, 
               filename: str, 
               key: Literal['data', 'modulated', 'reference', 'difference'] = 'data',
               ) -> None:
        """
        Export the data array to an fid file, following the format of a SIMPSON output. 

        Parameters
        ----------
        filename: str
            Name of the exported file. 
        key: str, optional
            Attribute name of which data to export. Default is 'data'. 
        
        Returns
        -------
        None

        Examples
        --------
        >>> fid.to_fid("output.fid")
        >>> fid.to_fid("output.fid", key='difference')       
        """
      
        _to_fid(filename, getattr(self, key))
    

    def to_txt(self, 
               filename: str, 
               key: Literal['data', 'modulated', 'reference', 'difference'] = 'data',
               order: Literal['C', 'F'] = 'C',
               fmt: str | list[str] ='%.6f', 
               ) -> None:
        """
        Export the data array to a txt file, following the format of a SIMPSON output. 

        Parameters
        ----------
        filename: str
            Name of the exported file. 
        key: str, optional
            Attribute name of which data to export. Default is 'data'. 
        order: str, optional
            Extra parameter for np.ndarray.reshape, defines whether the elements of the array are read in a C-like order \
                (same row first) or in a Fortran-like order (same column first). Default is 'C'. 
        fmt: str or sequence of strs, optional
            Extra parameter for np.savetxt, controls the format of the output. Check https://numpy.org/devdocs/reference/generated/numpy.savetxt.html \
                for more. Default is '%.6f'. 
        
        Returns
        -------
        None

        Examples
        --------
        >>> fid.to_txt("output.txt")
        >>> fid.to_txt("output.fid", key='difference', fmt='%.2f')       
        """

        _to_txt(filename, getattr(self, key), order=order, fmt=fmt)


@dataclass
class Fid_pair:
    """
    Template class for 1D intensity data with double-FID acquisition, such as REDOR, VT-DRENAR, DQ build-up. The input data \
        should take the form of a 1D array, the first half of the array being the modulated data, and the second half of the \
        data being the reference data. 

    Parameters
    ----------
    filename: str
        Input data file. 
    load_text_options: dict, optional
        Additional keyword arguments for np.loadtxt(). Default {}.

    Attributes
    ----------
    filename: str
        See Parameters.
    load_text_options: dict
        See Parameters. 

    data: np.ndarray
        Raw data read from the file, reshaped to (2, n_points) array.
    n_points: int
        Number of points in the experiment.
    modulated: np.ndarray
        Modulated signal (S', first FID), 1D array.
    reference: np.ndarray
        Reference signal (S₀, second FID), 1D array.
    difference: np.ndarray
        Normalized difference (1 - S'/S₀), 1D array.
    
    x_initial_value: float
        Initial value for x-axis. Default 0.
    loop_counter_start: int
        Starting value of loop counter.
    loop_counter_increment: int
        Increment of loop counter.
    length_per_counter: float
        Length of the real x-axis represented by each loop counter. Default None.
    num_continuous: int
        How many points is generated for an artificial x-axis used for trendlines. Default is 100. 
    
    x_discrete: np.ndarray
        Discrete x-axis values corresponding to the data points, 1D array.
    x_continuous: np.ndarray
        Continuous x-axis values for plotting the fitted curve, 1D array.
    loop_counters: np.ndarray
        Loop counter values, 1D array.

    Methods
    -------
    set_x_axis
        Create the x-axis corresponding to the data array. 
    calculate_difference
        Calculate the relative difference between the data array and a reference. 
    to_fid
        Export the data array to an fid file. 
    to_txt
        Export the data array to a txt file. 

    Examples
    --------
    >>> fid_pair = Fid_pair("redor_data.txt")
    >>> print(fid_pair.difference)
    >>> fid_pair.set_x_axis(x_initial_value=0, loop_counter_start=1, loop_counter_increment=1, length_per_counter=58.82, n_points=16)
    >>> fid_pair.to_txt("output_difference.txt", key='difference')
    """
    filename: str

    data: np.ndarray = field(init=False)
    n_points: int = field(init=False)
    modulated: np.ndarray = field(init=False)
    difference: np.ndarray = field(init=False)
    reference: np.ndarray = field(init=False)

    x_initial_value: float = field(init=False, default=0)
    loop_counter_start: int = field(init=False)
    loop_counter_increment: int = field(init=False)
    length_per_counter: float = field(init=False)
    num_continuous: int = field(init=False, default=100)

    x_discrete: np.ndarray = field(init=False)
    x_continuous: np.ndarray = field(init=False)
    loop_counters: np.ndarray = field(init=False)

    load_text_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            x_initial_value: float,
            loop_counter_start: int, 
            loop_counter_increment: int, 
            length_per_counter: float, 
            n_points: int, 
            num_continuous: int = 100,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create the x-axis corresponding to the data array. In the form of x_initial_value + length_per_counter*[list of \
            loop counter with defined start value, increment, and number of points]. 

        Parameters
        ----------
        x_initial_value: float
            Initial value for x-axis.
        loop_counter_start: int
            Starting value of loop counter.
        loop_counter_increment: int
            Increment of loop counter.
        length_per_counter: float
            Length of the real x-axis represented by each loop counter.
        n_points: int
            Number of points in the experiment.
        num_continuous: int, optional
            How many points is generated for an artificial x-axis used for trendlines. Default is 100. 

        Returns
        -------
        self.x_discrete: np.ndarray
            Discrete x-axis values for plotting experimental/simulated data.
        self.x_continuous: np.ndarray
            Artificial x-axis with (usually) more points used for trendlines. 
        self.loop_counters: np.ndarray
            Loop counter array in case needed. 

        Examples
        --------
        >>> fid_pair.set_x_axis(x_initial_value=0, loop_counter_start=1, loop_counter_increment=1, length_per_counter=58.82, n_points=16)
        """
        self.x_initial_value = x_initial_value
        self.loop_counter_start = loop_counter_start
        self.loop_counter_increment = loop_counter_increment
        self.length_per_counter = length_per_counter
        self.n_points = n_points
        self.num_continuous = num_continuous

        self.x_discrete, self.x_continuous, self.loop_counters = _set_x_axis(x_initial_value, 
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
        """
        Calculate the reference and difference array from the data. 

        Parameters
        ----------
        modulated: np.ndarray
            Modulated signal (S'), 1D array.
        reference: np.ndarray
            Reference signal (S₀), 1D array.

        Returns
        -------
        self.difference: np.ndarray
            Array of the relative differences between the modulated data and reference. 
        """
        self.modulated = modulated
        self.reference = reference 
        self.difference = _calculate_difference(modulated, reference)

        return self.difference

    
    def to_fid(self, 
               filename: str, 
               key: Literal['data', 'modulated', 'reference', 'difference'] = 'data',
               ) -> None:
        """
        Export the data array to an fid file, following the format of a SIMPSON output. 

        Parameters
        ----------
        filename: str
            Name of the exported file. 
        key: str, optional
            Attribute name of which data to export. Default is 'data'. 
        
        Returns
        -------
        None

        Examples
        --------
        >>> fid_pair.to_fid("output.fid")
        >>> fid_pair.to_fid("output.fid", key='difference')       
        """
        _to_fid(filename, getattr(self, key))
    

    def to_txt(self, 
               filename: str, 
               key: Literal['data', 'modulated', 'reference', 'difference'] = 'data',
               order: Literal['C', 'F'] = 'C',
               fmt: str ='%.6f', 
               ) -> None:
        """
        Export the data array to a txt file, following the format of a SIMPSON output. 

        Parameters
        ----------
        filename: str
            Name of the exported file. 
        key: str, optional
            Attribute name of which data to export. Default is 'data'. 
        order: str, optional
            Extra parameter for np.ndarray.reshape, defines whether the elements of the array are read in a C-like order \
                (same row first) or in a Fortran-like order (same column first). Default is 'C'. 
        fmt: str or sequence of strs, optional
            Extra parameter for np.savetxt, controls the format of the output. Check https://numpy.org/devdocs/reference/generated/numpy.savetxt.html \
                for more. Default is '%.6f'. 
        
        Returns
        -------
        None

        Examples
        --------
        >>> fid_pair.to_txt("output.txt")
        >>> fid_pair.to_txt("output.fid", key='difference', fmt='%.2f')       
        """
        _to_txt(filename, getattr(self, key), order=order, fmt=fmt)


@dataclass
class Fid_triple:
    """
    Template class for 1D intensity data with triple-FID acquisition, such as compensated REDOR. The compensated difference \
        is calculated as uncompensated difference + alpha * difference between compensated and reference signal. The input data \
        should take the form of a 1D array, the first 1/3 of the array being the modulated data, the second 1/3 being the \
        additional compensated data, te last 1/3 being the reference data. 
    
    Parameters
    ----------
    filename: str
        Input data file.
    alpha: float, optional
        Compensation factor for triple-FID acquisition. Default 1.
    load_text_options: dict, optional
        Additional keyword arguments for np.loadtxt(). Default {}.

    Attributes
    ----------
    filename: str
        See Parameters.
    alpha: float
        See Parameters.
    load_text_options: dict
        See Parameters.
    
    data: np.ndarray
        Raw data read from the file, reshaped to (3, n_points) array.
    n_points: int
        Number of points in the experiment.
    modulated: np.ndarray
        Modulated signal (S', first FID), 1D array.
    compensated: np.ndarray
        Compensated signal (S*, second FID), 1D array.
    reference: np.ndarray
        Reference signal (S₀, third FID), 1D array.
    difference: np.ndarray
        Normalized difference with compensation ((1 - S'/S₀) + alpha*(1 - S*/S₀)), 1D array.
    difference_not_compensated: np.ndarray
        Normalized difference without compensation (1 - S'/S₀), 1D array.
    
    x_initial_value: float
        Initial value for x-axis. Default 0.
    loop_counter_start: int
        Starting value of loop counter.
    loop_counter_increment: int
        Increment of loop counter.
    length_per_counter: float
        Length per loop counter.
    num_continuous: int
        Number of points for continuous x-axis. Default 100. 
    
    x_discrete: np.ndarray
        Discrete x-axis values corresponding to the data points, 1D array.
    x_continuous: np.ndarray
        Continuous x-axis values for plotting the fitted curve, 1D array.
    loop_counters: np.ndarray
        Loop counter values, 1D array.

    Methods
    -------
    set_x_axis
        Create the x-axis corresponding to the data array. 
    calculate_difference
        Calculate the relative difference between the data array and a reference. 
    to_fid
        Export the data array to an fid file. 
    to_txt
        Export the data array to a txt file. 
    
    Examples
    --------
    >>> fid_triple = Fid_triple("compensated_redor.txt", alpha=1.0)
    >>> print(fid_triple.difference)
    >>> fid_triple.set_x_axis(x_initial_value=0, loop_counter_start=1, loop_counter_increment=1, length_per_counter=58.82, n_points=16)
    >>> fid_triple.to_txt("output_difference.txt", key='difference')
    """
    filename: str
    alpha: float = 1

    data: np.ndarray = field(init=False)
    n_points: int = field(init=False)
    modulated: np.ndarray = field(init=False)
    compensated: np.ndarray = field(init=False)
    difference: np.ndarray = field(init=False)
    difference_not_compensated: np.ndarray = field(init=False)
    reference: np.ndarray = field(init=False)

    x_initial_value: float = field(init=False, default=0)
    loop_counter_start: int = field(init=False)
    loop_counter_increment: int = field(init=False)
    length_per_counter: float = field(init=False)
    num_continuous: int = field(init=False, default=100)

    x_discrete: np.ndarray = field(init=False)
    x_continuous: np.ndarray = field(init=False)
    loop_counters: np.ndarray = field(init=False)

    load_text_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if PurePath(self.filename).suffix == '.txt':
            self.data = np.loadtxt(self.filename, **self.load_text_options)
        elif PurePath(self.filename).suffix == '.fid':
            self.data, _ = read_fid(self.filename)
        else:
            raise TypeError("File must be .txt or .fid! ")
        
        if self.data.shape[0] % 3 == 1:
            raise ValueError("Data length is not divisible by 3, cannot be reshaped into triples.")
        self.data = np.reshape(self.data, (3, -1))
        self.n_points = self.data.shape[1]
        self.modulated = self.data[0]
        self.compensated = self.data[1]
        self.reference = self.data[2]

        self.calculate_difference(self.modulated, self.compensated, self.reference)


    def set_x_axis(self, 
            x_initial_value: float,
            loop_counter_start: int, 
            loop_counter_increment: int, 
            length_per_counter: float, 
            n_points: int, 
            num_continuous: int = 100,
            ) -> None:
        """
        Create the x-axis corresponding to the data array. In the form of x_initial_value + length_per_counter*[list of \
            loop counter with defined start value, increment, and number of points]. 

        Parameters
        ----------
        x_initial_value: float
            Initial value for x-axis.
        loop_counter_start: int
            Starting value of loop counter.
        loop_counter_increment: int
            Increment of loop counter.
        length_per_counter: float
            Length of the real x-axis represented by each loop counter.
        n_points: int
            Number of points in the experiment.
        num_continuous: int, optional
            How many points is generated for an artificial x-axis used for trendlines. Default is 100. 

        Returns
        -------
        self.x_discrete: np.ndarray
            Discrete x-axis values for plotting experimental/simulated data.
        self.x_continuous: np.ndarray
            Artificial x-axis with (usually) more points used for trendlines. 
        self.loop_counters: np.ndarray
            Loop counter array in case needed. 

        Examples
        --------
        >>> fid_triple.set_x_axis(x_initial_value=0, loop_counter_start=1, loop_counter_increment=1, length_per_counter=58.82, n_points=16)
        """
        self.x_initial_value = x_initial_value
        self.loop_counter_start = loop_counter_start
        self.loop_counter_increment = loop_counter_increment
        self.length_per_counter = length_per_counter
        self.n_points = n_points
        self.num_continuous = num_continuous

        self.x_discrete, self.x_continuous, self.loop_counters = _set_x_axis(x_initial_value, 
                                                            loop_counter_start, 
                                                            loop_counter_increment, 
                                                            length_per_counter, 
                                                            n_points, 
                                                            num_continuous)


    def calculate_difference(self, 
                             modulated: np.ndarray,
                             compensated: np.ndarray,
                             reference: np.ndarray, 
                             ) -> np.ndarray:
        """
        Calculate the reference and difference array from the data. 

        Parameters
        ----------
        modulated: np.ndarray
            Modulated signal (S'), 1D array.
        compensated: np.ndarray
            Compensated signal (S*), 1D array.
        reference: np.ndarray
            Reference signal (S₀), 1D array.

        Returns
        -------
        self.difference: np.ndarray
            Array of the compensated differences. 
        """
        self.modulated = modulated
        self.compensated = compensated
        self.reference = reference 

        self.difference_not_compensated = _calculate_difference(modulated, reference)
        self.difference = self.difference_not_compensated + self.alpha * _calculate_difference(compensated, reference)

        return self.difference

    
    def to_fid(self, 
               filename: str, 
               key: Literal['data', 'modulated', 'reference', 'compensated', 'difference_not_compensated', 'difference'] = 'data',
               ) -> None:
        """
        Export the data array to an fid file, following the format of a SIMPSON output. 

        Parameters
        ----------
        filename: str
            Name of the exported file. 
        key: str, optional
            Attribute name of which data to export. Default is 'data'. 
        
        Returns
        -------
        None

        Examples
        --------
        >>> fid_triple.to_fid("output.fid")
        >>> fid_triple.to_fid("output.fid", key='difference')       
        """
        _to_fid(filename, getattr(self, key))
    

    def to_txt(self, 
               filename: str, 
               key: Literal['data', 'modulated', 'reference', 'compensated', 'difference_not_compensated', 'difference'] = 'data',
               order: Literal['C', 'F'] = 'C',
               fmt: str ='%.6f', 
               ) -> None:
        """
        Export the data array to a txt file, following the format of a SIMPSON output. 

        Parameters
        ----------
        filename: str
            Name of the exported file. 
        key: str, optional
            Attribute name of which data to export. Default is 'data'. 
        order: str, optional
            Extra parameter for np.ndarray.reshape, defines whether the elements of the array are read in a C-like order \
                (same row first) or in a Fortran-like order (same column first). Default is 'C'. 
        fmt: str or sequence of strs, optional
            Extra parameter for np.savetxt, controls the format of the output. Check https://numpy.org/devdocs/reference/generated/numpy.savetxt.html \
                for more. Default is '%.6f'. 
        
        Returns
        -------
        None

        Examples
        --------
        >>> fid_triple.to_txt("output.txt")
        >>> fid_triple.to_txt("output.fid", key='difference', fmt='%.2f')       
        """
        _to_txt(filename, getattr(self, key), order=order, fmt=fmt)


@dataclass
class CTDrenar(Fid_single):
    """
    Class for handling CT-DRENAR data, a single-FID acquisition experiment with phase incrementation. The input data is a txt \
        or SIMPSON fid file with a 1D data array. 

    References
    ----------
    Ren, J. & Eckert, H. (2000). Measurement of homonuclear magnetic dipole-dipole interactions 
    in multiple 1/2-spin systems using constant-time DQ-DRENAR NMR. Journal of Magnetic Resonance, 260(1), 46-53.
    https://doi.org/10.1016/j.jmr.2015.08.022

    Parameters
    ----------
    filename: str
        Input data file.
    phase_range: tuple[float, float], optional
        Phase range (min_phase, max_phase) in degrees. Default is None.
    phase_increment: float, optional
        Increment of phase in degrees. Default is None.
    reference_idx: int, optional
        Index of the reference point in the data. Default is None (and the middle point of the data is taken if n_points is odd).
    l0: int, optional
        Rotor cycles for the first point. Default is None.
    spin_rate: float, optional
        Spinning rate in kHz. Default is None.
    num_continuous: int, optional
        Number of points for continuous x-axis. Default is 100.
    verbose: bool, optional
        Whether to print detailed information during initialization. Default is False.
    load_text_options: dict, optional
        Additional keyword arguments for np.loadtxt(). Default {}.

    Attributes
    ----------
    filename: str
        See Parameters.
    load_text_options: dict
        See Parameters.
    verbose: bool
        See Parameters.
    
    phase_range: tuple[float, float]
        See Parameters.
    phase_increment: float
        See Parameters.

    reference_idx: int
        See Parameters.
    l0: int
        See Parameters.
    spin_rate: float
        See Parameters.
    num_continuous: int
        See Parameters.
    
    data: np.ndarray
        Raw data read from the file corresponding to the modulated signal (S'), 1D array.
    reference: np.ndarray
        Reference signal (S₀), 1D array.
    difference: np.ndarray
        Normalized DQ intensity (1 - S'/S₀), 1D array.
    n_points: int, optional
        Number of points in the experiment.

    phase_discrete: np.ndarray
        Discrete phase values in degrees, 1D array.
    phase_continuous: np.ndarray
        Continuous phase values for plotting the fitted curve, in degrees, 1D array.

    dephasing_time: float
        Dephasing time calculated from l0 and spin rate (if provided), in ms.
    popt: np.ndarray
        Optimized parameters from curve fitting.
    pconv: np.ndarray
        Covariance matrix from curve fitting.
    z_opt: float
        Optimized z-value from fitting, in ms².
    beff_opt: float
        Effective dipolar coupling constant from fitting, in kHz.
    predict: np.ndarray
        Fitted curve prediction values on phase_continuous.
    
    Methods
    -------
    plot_difference
        Plot the relative difference (1 - S'/S₀) against the phase angle.
    fit
        Fit CT-DRENAR data with the analytical function and calculate the effective dipolar coupling constant
    plot_fit
        Plot the fitted curve against the experimental data.

    Examples
    --------
    >>> exp = CTDrenar("ct_drenar.txt", l0=2, spin_rate=17)
    >>> print(exp.difference)
    >>> fig, ax = exp.plot_difference()
    >>> beff = exp.fit()
    >>> fig, ax = exp.plot_fit(show_legend=True)
    """
    phase_range: tuple[float, float] | None = None
    phase_increment: float | None = None

    l0: int | None = None
    spin_rate: float | None = None
    num_continuous: int = 100
    verbose: bool = False
    
    phase_discrete: np.ndarray = field(init=False)
    phase_continuous: np.ndarray = field(init=False)

    dephasing: np.ndarray = field(init=False)
    dephasing_time: float = field(init=False)
    popt: np.ndarray = field(init=False)
    pconv: np.ndarray = field(init=False)
    z_opt: float = field(init=False)
    beff_opt: float = field(init=False)
    predict: np.ndarray = field(init=False)
    

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.reference_idx is None:
            if self.n_points % 2 == 0:
                warnings.warn("Even number of points, the reference point cannot be found.")
            else:
                self.reference_idx = self.n_points // 2   # python index, starting from 0
                self.calculate_difference(self.reference_idx)

        self.phase_range, self.phase_increment = _set_phases(self.phase_range, self.phase_increment, self.n_points)
        self.phase_discrete = np.linspace(self.phase_range[0], self.phase_range[1], num=self.n_points)
        self.phase_continuous = np.linspace(self.phase_range[0], self.phase_range[1], num=100)

        self.phase_discrete, self.phase_continuous, _ = self.set_x_axis(self.phase_range[0], 
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
        Plot the relative difference (1 - S'/S₀) against the phase angle.

        Parameters
        ----------
        xlim: tuple[float, float], optional
            Limits for x-axis
        ylim: tuple[float, float], optional
            Limits for y-axis
        figure_size: tuple[float, float], optional
            Size of the figure (width, height)
        show_legend: bool, optional
            Whether to show legend
        **kwargs: optional
            Additional keyword arguments for plt.plot()

        Returns
        -------
        fig: Figure
            The created figure object
        ax: Axes
            The created axes object
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

        Parameters
        ----------
        function: Callable, optional
            The function to fit the data (default: ctdrenar)

        Returns
        -------
        beff_opt: float
            Effective dipolar coupling constant calculated from the optimized z-value, in kHz, if the dephasing time is provided. 
        z_opt: float
            Optimized z-value from the fitting, in ms^2, if dephasing time is not provided. 
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

        Parameters
        ----------
        xlim: tuple, optional
            Limits for x-axis. Default is None. 
        ylim: tuple, optional
            Limits for y-axis. Default is None.
        figure_size: tuple, optional
            Size of the figure (width, height). Default is (4, 4).
        color: str, optional
            Color for both experimental data and fitted curve. Default is '#0092c8'. 
        show_legend: bool, optional
            Whether to show legend. Default is False.
        
        Returns
        -------
        fig: Figure
            The created figure object
        ax: Axes
            The created axes object
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
    Class for handling REDOR data, a double-FID acquisition experiment with time incrementation.

    Parameters
    ----------
    filename: str
        Input data file.
    l0: int, optional
        Rotor cycles for the first point. Default None.
    l10: int, optional
        Increment constant defined in the pulse program. Default None.
    spin_rate: float, optional
        Spinning rate in kHz. Default None.
    gamma_I: float, optional
        Gyromagnetic ratio of the observed nucleus in MHz/T. Default None.
    gamma_S: float, optional
        Gyromagnetic ratio of the dephasing nucleus in MHz/T. Default None.
    truncated_n_points: int, optional
        Truncated number of points to use. Default None.
    num_continuous: int, optional
        Number of points for continuous x-axis. Default 100.
    verbose: bool, optional
        Whether to print detailed information during initialization. Default False.
    load_text_options: dict, optional
        Additional keyword arguments for np.loadtxt(). Default {}.

    Attributes
    ----------
    filename: str
        See Parameters.
    l0: int
        See Parameters.
    l10: int
        See Parameters.
    spin_rate: float
        See Parameters.
    gamma_I: float
        See Parameters.
    gamma_S: float
        See Parameters.
    truncated_n_points: int
        See Parameters.
    num_continuous: int
        See Parameters.
    verbose: bool
        See Parameters.
    load_text_options: dict
        See Parameters.

    data: np.ndarray
        Raw data read from the file, reshaped to (2, n_points) array.
    n_points: int
        Number of data points in the experiment.
    dephasing: np.ndarray
        Dephasing signal (S'), 1D array.
    reference: np.ndarray
        Reference signal (S₀), 1D array.
    difference: np.ndarray
        Normalized difference (1 - S'/S₀), 1D array.
    
    time_discrete: np.ndarray
        Discrete time values corresponding to data points, in ms, 1D array.
    time_continuous: np.ndarray
        Continuous time values for plotting the fitted curve, in ms, 1D array.
    loop_counters: np.ndarray
        Loop counter values, 1D array.
    
    popt: np.ndarray
        Optimized parameters from curve fitting.
    pconv: np.ndarray
        Covariance matrix from curve fitting.
    z_opt: float
        Optimized z-value from fitting, in kHz².
    d_opt: float
        Optimized d-value from fitting.
    beff_opt: float
        Effective dipolar coupling constant from fitting, in kHz.
    r_opt: float
        Optimized distance from fitting, in Å.
    predict: np.ndarray
        Fitted curve prediction values on time_continuous.

    Methods
    -------
    set_time_axis
        Create the time axis corresponding to the data array using rotor cycle parameters.
    plot_difference
        Plot the normalized difference (1 - S'/S₀) against the recoupling time.
    fit
        Fit REDOR data with the analytical function and calculate the effective dipolar coupling constant.
    plot_fit
        Plot the fitted curve together with the experimental data.

    Examples
    --------
    >>> exp = Redor("redor_data.txt", l0=1, l10=1, spin_rate=17.0, truncated_n_points=8)
    >>> print(exp.difference)
    >>> fig, ax = exp.plot_difference()
    >>> exp.set_time_axis(l0, l10, spin_rate)
    >>> exp.gamma_I, exp.gamma_S = gamma.C, gamma.P
    >>> beff, r = exp.fit()
    >>> fig, ax = exp.plot_fit(show_legend=True)
    """
    l0: int | None = None
    l10: int | None = None
    spin_rate: float | None = None

    gamma_I: float | None = None
    gamma_S: float | None = None

    truncated_n_points: int | None = None
    num_continuous: int = 100
    verbose: bool = False

    time_discrete: np.ndarray = field(init=False)
    time_continuous: np.ndarray = field(init=False)

    dephasing: np.ndarray = field(init=False)
    reference: np.ndarray = field(init=False)
    difference: np.ndarray = field(init=False)

    popt: np.ndarray = field(init=False)
    pconv: np.ndarray = field(init=False)
    z_opt: float = field(init=False)
    d_opt: float = field(init=False)
    beff_opt: float = field(init=False)
    r_opt: float = field(init=False)
    predict: np.ndarray = field(init=False)

    def __post_init__(self):

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
            
        elif PurePath(self.filename).suffix == '.fid':  # assert simulated fid using simpson with the first point being the reference. 
            self.data, _ = read_fid(self.filename)
            if self.truncated_n_points is None:
                self.n_points = np.shape(self.data)[0] - 1   
            else:
                self.n_points = self.truncated_n_points     
            
            # if the following truncation is not correct, you need to use the original contents in the data attribute. 
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
            self.set_time_axis(self.l0, self.l10, self.spin_rate, self.num_continuous)
            # For different pulse sequences, the time axis can also be defined by calling the .set_x_axis() method using more general loop counter settings. 
    

    # I'm not sure whether to keep these properties yet. 
    @property
    def x_initial_value(self) -> float:  # type: ignore
        return 0

    @property
    def loop_counter_start(self) -> int | None:  # type: ignore
        return self.l0 + 1 if self.l0 is not None else None
    
    @property
    def loop_counter_increment(self) -> int | None:  # type: ignore
        return self.l10 * 2 if self.l10 is not None else None
    
    @property
    def length_per_counter(self) -> float | None:  # type: ignore
        return 1/self.spin_rate if self.spin_rate is not None else None


    def set_time_axis(self, 
        l0: int, 
        l10: int, 
        spin_rate: float, 
        num_continuous: int = 100,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create the time axis corresponding to the data array. In the form of initial_value + (1/spin_rate)*[list of \
            loop counter with defined start value (l0 + 1), increment (l10 * 2), and number of points].

        Parameters
        ----------
        l0: int
            Rotor cycles for the first point.
        l10: int
            Increment constant defined in the pulse program.
        spin_rate: float
            Spinning rate in kHz.
        num_continuous: int, optional
            How many points is generated for an artificial x-axis used for trendlines. Default is 100.

        Returns
        -------
        self.time_discrete: np.ndarray
            Discrete time values for plotting experimental/simulated data, in ms.
        self.time_continuous: np.ndarray
            Artificial time axis with (usually) more points used for trendlines, in ms.
        self.loop_counters: np.ndarray
            Loop counter array in case needed.

        Examples
        --------
        >>> exp.set_time_axis(l0=1, l10=1, spin_rate=17.0)
        """
        self.l0 = l0
        self.l10 = l10
        self.spin_rate = spin_rate
        self.num_continuous = num_continuous

        self.time_discrete, self.time_continuous, self.loop_counters = _set_x_axis(0, 
                            self.l0 + 1, 
                            self.l10 * 2, 
                            1/self.spin_rate, 
                            self.n_points, 
                            self.num_continuous)
        
        print(f"\n{self.n_points :d} steps, time increment {2*self.l10} rotor cycles, {self.time_discrete[1] - self.time_discrete[0]:.3f} ms for each step.") if self.verbose else ""
        
        return self.time_discrete, self.time_continuous, self.loop_counters


    def plot_difference(self, 
                        xlim: tuple[float, float] | None = None, 
                        ylim: tuple[float, float] | None = None, 
                        figure_size: tuple[float, float] = (4, 4),
                        show_legend: bool = False, 
                        **kwargs: Any
                        ) -> tuple[Figure, Axes]:
        """
        Plot the relative difference (1 - S'/S₀) against the recoupling time.

        Parameters
        ----------
        xlim: tuple[float, float], optional
            Limits for x-axis. Default is None.
        ylim: tuple[float, float], optional
            Limits for y-axis. Default is None.
        figure_size: tuple[float, float], optional
            Size of the figure (width, height). Default is (4, 4).
        show_legend: bool, optional
            Whether to show legend. Default is False.
        **kwargs: optional
            Additional keyword arguments for plt.plot()

        Returns
        -------
        fig: Figure
            The created figure object
        ax: Axes
            The created axes object

        Examples
        --------
        >>> fig, ax = exp.plot_difference()
        >>> fig, ax = exp.plot_difference(xlim=(0, 10), ylim=(0, 0.3))
        """
        if self.time_discrete is None:
            if isinstance(self.l0, Number) and isinstance(self.l10, Number) and isinstance(self.spin_rate, Number):
                self.set_time_axis(self.l0, self.l10, self.spin_rate, self.num_continuous)
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

        Parameters
        ----------
        function: Callable, optional
            The function to fit the data. Default is narumii.functions.redor_bessel(5).

        Returns
        -------
        beff_opt: float
            Effective dipolar coupling constant calculated from the optimized parameter, in kHz.
        r_opt: float or None
            Optimized distance from the fitting, in Å, if gamma_I and gamma_S are provided. Otherwise None.

        Examples
        --------
        >>> beff, r = exp.fit()
        >>> print(f'Effective coupling: {beff:.3f} kHz, Distance: {r:.3f} Å')
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
        Plot the fitted curve together with the experimental data.

        Parameters
        ----------
        xlim: tuple[float, float], optional
            Limits for x-axis. Default is None.
        ylim: tuple[float, float], optional
            Limits for y-axis. Default is None.
        figure_size: tuple[float, float], optional
            Size of the figure (width, height). Default is (4, 4).
        color: str, optional
            Color for both experimental data and fitted curve. Default is '#0092c8'.
        show_legend: bool, optional
            Whether to show legend. Default is False.

        Returns
        -------
        fig: Figure
            The created figure object
        ax: Axes
            The created axes object

        Examples
        --------
        >>> fig, ax = exp.plot_fit(show_legend=True)
        >>> fig, ax = exp.plot_fit(xlim=(0, 10), ylim=(0, 0.3), color='red')
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
    Class for handling compensated REDOR data, a triple-FID acquisition experiment with time incrementation.

    References
    ----------
    Gullion, T., & Schaefer, J. (2000). Eliminating artifacts from baseline distortion in 
    rotational-echo double-resonance NMR. Journal of Magnetic Resonance, 144(1), 174-180.
    https://doi.org/10.1006/jmre.2000.2191

    Parameters
    ----------
    filename: str
        Input data file.
    alpha: float, optional
        Compensation factor defined in the pulse program. Default 1.
    l0: int, optional
        Rotor cycles for the first point. Default None.
    l10: int, optional
        Increment constant defined in the pulse program. Default None.
    spin_rate: float, optional
        Spinning rate in kHz. Default None.
    gamma_I: float, optional
        Gyromagnetic ratio of the observed nucleus in MHz/T. Default None.
    gamma_S: float, optional
        Gyromagnetic ratio of the dephasing nucleus in MHz/T. Default None.
    truncated_n_points: int, optional
        Truncated number of points to use. Default None.
    num_continuous: int, optional
        Number of points for continuous x-axis. Default 100.
    verbose: bool, optional
        Whether to print detailed information during initialization. Default False.
    load_text_options: dict, optional
        Additional keyword arguments for np.loadtxt(). Default {}.

    Attributes
    ----------
    filename: str
        See Parameters.
    alpha: float
        See Parameters.
    l0: int
        See Parameters.
    l10: int
        See Parameters.
    spin_rate: float
        See Parameters.
    gamma_I: float
        See Parameters.
    gamma_S: float
        See Parameters.
    truncated_n_points: int
        See Parameters.
    num_continuous: int
        See Parameters.
    verbose: bool
        See Parameters.
    load_text_options: dict
        See Parameters.
    
    data: np.ndarray
        Raw data read from the file, reshaped to (3, n_points) array.
    n_points: int
        Number of data points in the experiment.
    dephasing: np.ndarray
        Dephasing signal (S'), 1D array.
    compensation: np.ndarray
        Compensation signal (S*), 1D array.
    reference: np.ndarray
        Reference signal (S₀), 1D array.
    difference: np.ndarray
        Compensated difference (1 - S'/S₀ + α(1 - S*/S₀)), 1D array.
    
    time_discrete: np.ndarray
        Discrete time values corresponding to data points, in ms, 1D array.
    time_continuous: np.ndarray
        Continuous time values for plotting the fitted curve, in ms, 1D array.
    loop_counters: np.ndarray
        Loop counter values, 1D array.
    
    popt: np.ndarray
        Optimized parameters from curve fitting.
    pconv: np.ndarray
        Covariance matrix from curve fitting.
    z_opt: float
        Optimized z-value from fitting, in kHz².
    d_opt: float
        Optimized d-value from fitting.
    beff_opt: float
        Effective dipolar coupling constant from fitting, in kHz.
    r_opt: float
        Optimized distance from fitting, in Å.
    predict: np.ndarray
        Fitted curve prediction values on time_continuous.

    Methods
    -------
    set_time_axis
        Create the time axis corresponding to the data array using rotor cycle parameters.

    Examples
    --------
    >>> exp = Redor3("compensated_redor.txt", alpha=1.0, l0=1, l10=1, spin_rate=17.0)
    >>> fig, ax = exp.plot_difference()
    >>> exp.set_time_axis(0, l0 + 1, l10 * 2, 1/spin_rate, n_points)
    >>> exp.gamma_I, exp.gamma_S = narumii.gamma.C, narumii.gamma.P
    >>> beff, r = exp.fit()
    >>> fig, ax = exp.plot_fit(show_legend=True)
    """
    l0: int | None = None
    l10: int | None = None
    spin_rate: float | None = None

    gamma_I: float | None = None
    gamma_S: float | None = None

    truncated_n_points: int | None = None
    num_continuous: int = 100
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

        elif PurePath(self.filename).suffix == '.fid':  # assert simulated fid using simpson with the first point being the reference. 
            self.data, _ = read_fid(self.filename)
            if self.truncated_n_points is None:
                self.n_points = np.shape(self.data)[0] - 1   
            else:
                self.n_points = self.truncated_n_points     
            
            # if the following truncation is not correct, you need to use the original contents in the data attribute. 
            self.dephasing = self.data[1:self._n_points+1]
            self.compensation = np.ones(self._n_points) * self.data[0]
            self.reference = self.compensation

        else:
            raise TypeError("File must be .txt or .fid! ")      
            
        if np.any(self.reference == 0):
            raise ValueError("Reference contains zero(s), cannot divide.")
        else:
            self.difference = 1 - self.dephasing/self.reference + self.alpha - self.alpha*self.compensation/self.reference

        print("\nList of data: ") if self.verbose else ""
        print(self.data) if self.verbose else ""
        print("\nList of differences: ") if self.verbose else ""
        print(self.difference) if self.verbose else ""
        
        if isinstance(self.l0, Number) and isinstance(self.l10, Number) and isinstance(self.spin_rate, Number):
            self.set_time_axis(self.l0, self.l10, self.spin_rate, self.num_continuous)
            # For different pulse sequences, the time axis can also be defined by calling the .set_x_axis() method using more general loop counter settings. 
    

    def set_time_axis(self, 
        l0: int, 
        l10: int, 
        spin_rate: float, 
        num_continuous: int = 100,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create the time axis corresponding to the data array. In the form of initial_value + (1/spin_rate)*[list of \
            loop counter with defined start value (l0 + 1), increment (l10 * 2), and number of points].

        Parameters
        ----------
        l0: int
            Rotor cycles for the first point.
        l10: int
            Increment constant defined in the pulse program.
        spin_rate: float
            Spinning rate in kHz.
        num_continuous: int, optional
            How many points is generated for an artificial x-axis used for trendlines. Default is 100.

        Returns
        -------
        self.time_discrete: np.ndarray
            Discrete time values for plotting experimental/simulated data, in ms.
        self.time_continuous: np.ndarray
            Artificial time axis with (usually) more points used for trendlines, in ms.
        self.loop_counters: np.ndarray
            Loop counter array in case needed.

        Examples
        --------
        >>> exp = Redor3("compensated_redor.txt", alpha=1.0)
        >>> exp.set_time_axis(l0=1, l10=1, spin_rate=17.0)
        """
        self.l0 = l0
        self.l10 = l10
        self.spin_rate = spin_rate
        self.num_continuous = num_continuous

        self.time_discrete, self.time_continuous, self.loop_counters = _set_x_axis(0, 
                            self.l0 + 1, 
                            self.l10 * 2, 
                            1/self.spin_rate, 
                            self.n_points, 
                            self.num_continuous)
        
        print(f"\n{self.n_points :d} steps, time increment {2*self.l10} rotor cycles, {self.time_discrete[1] - self.time_discrete[0]:.3f} ms for each step.") if self.verbose else ""
        
        return self.time_discrete, self.time_continuous, self.loop_counters


@dataclass
class DoubleQuantum(Fid_pair): 
    """
    Class for handling double quantum build-up data, a double-FID acquisition experiment with time incrementation.

    Parameters
    ----------
    filename: str
        Input data file.
    l0: int, optional
        Rotor cycles for the first point. Default None.
    l10: int, optional
        Increment constant defined in the pulse program. Default None.
    spin_rate: float, optional
        Spinning rate in kHz. Default None.
    num_continuous: int, optional
        Number of points for continuous x-axis. Default 100.
    verbose: bool, optional
        Whether to print detailed information during initialization. Default False.
    load_text_options: dict, optional
        Additional keyword arguments for np.loadtxt(). Default {}.

    Attributes
    ----------
    filename: str
        See Parameters.
    l0: int
        See Parameters.
    l10: int
        See Parameters.
    spin_rate: float
        See Parameters.
    num_continuous: int
        See Parameters.
    verbose: bool
        See Parameters.
    load_text_options: dict
        See Parameters.

    data: np.ndarray
        Raw data read from the file, reshaped to (2, n_points) array.
    modulated: np.ndarray
        Modulated DQ signal (first FID), 1D array.
    reference: np.ndarray
        Reference signal (second FID), 1D array.
    difference: np.ndarray
        Normalized DQ build-up (1 - S/S₀), 1D array.
    n_points: int
        Number of data points.
    time_discrete: np.ndarray
        Discrete time values corresponding to data points, in ms, 1D array. Only present if l0, l10, and spin_rate are provided.
    time_continuous: np.ndarray
        Continuous time values for plotting the fitted curve, in ms, 1D array. Only present if l0, l10, and spin_rate are provided.
    loop_counters: np.ndarray
        Loop counter values, 1D array. Only present if l0, l10, and spin_rate are provided.

    Examples
    --------
    >>> exp = DoubleQuantum("dq_buildup.txt", l0=1, l10=1, spin_rate=10.0)
    >>> fig, ax = exp.plot_difference()
    """
    l0: int | None = None
    l10: int | None = None
    spin_rate: float | None = None

    verbose: bool = False

    def __post_init__(self):

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