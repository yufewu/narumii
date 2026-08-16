import numpy as np
import matplotlib.pyplot as plt
from typing import TextIO


def read_fid(
        filename: str
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads a SIMPSON fid file and extracts the real and imaginary parts of the data.
    
    Parameters
    ----------
    filename: str
        The path to the fid file to be read.

    Returns
    -------
    data_real: np.ndarray
        The real part of the data extracted from the fid file.
    data_imag: np.ndarray
        The imaginary part of the data extracted from the fid file.
    """
    
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


def read_results(
        filename: str, 
        skiprows: int = 1
        ) -> np.ndarray:
    """
    Reads a results file and returns the data as a numpy array.

    Parameters
    ----------
    filename : str
        The path to the results file to be read.
    skiprows : int, optional
        The number of rows to skip at the beginning of the file. Default is 1.

    Returns
    -------
    results: np.ndarray
        The data read from the results file.
    """
    
    results = np.loadtxt(filename, delimiter=",", skiprows=skiprows)
    results = results.transpose()

    return results


class Tee:
    """
    Redirects output to multiple file-like objects.

    Parameters
    ----------
    *files: file-like objects
        The file-like objects to which output will be written.
    
    Methods
    -------
    write
        Write to the outputs.
    flush
        Flush the outputs. 

    Examples
    --------
    >>> sys.stdout = Tee(sys.stdout, f); print(...); sys.stdout = sys.__stdout__
    """
    def __init__(
        self, 
        *files: TextIO
        ) -> None:
        self.files = files
    
    def write(
        self, 
        obj: str
        ) -> None:
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self) -> None:
        for f in self.files:
            f.flush()


def find_uncertainty_range(basename: str, 
                            angles_dir: str, 
                            num_angle_sets: int = 5, 
                            ) -> None: 
    """
    Find the uncertainty range of the fitted parameter from a series of fitting results. 

    Parameters
    ----------
    basename: str
        The base name of the results files.
    angles_dir: str
        The directory containing the angle sets.
    num_angle_sets: int, optional
        The number of angle sets to consider. Default is 5.
    
    Returns
    -------
    None

    Examples
    --------
    >>> find_uncertainty_range(basename="sample_fit", angles_dir="./", num_angle_sets=5)
    """
    b_values = np.array([])
    results_files = [f"./{basename}_angles{i+1}_fit.results" for i in range(num_angle_sets)]
    for results_file in results_files:
        _b, _ = read_results(results_file)
        b_values = np.append(b_values, np.absolute(_b))

    serials = np.arange(1, len(b_values) + 1)
    fig, ax = plt.subplots(figsize=(5.6, 4), constrained_layout=True)
    ax.scatter(serials, b_values)

    ax.set_xlabel('Simulation serial number', fontsize=12)
    ax.set_ylabel(r"$b_\mathrm{app}$ (Hz)", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

    print(f"Maximum b value: {np.max(b_values)}")
    print(f"Minimum b value: {np.min(b_values)}")

    angles_files = [f"{angles_dir}/angles_{i+1}.txt" for i in range(num_angle_sets)]
    angle_sets = []
    for angles_file in angles_files:
        with open(angles_file, 'r') as f:
            for line in f:
                angle_set = [float(x.strip()) for x in line.split(',')]
                angle_sets.append(angle_set)

    unique_angle_sets, unique_indices = np.unique(angle_sets, axis=0, return_index=True)
    unique_b_values = b_values[unique_indices]

    sorted_indices = np.argsort(unique_b_values)
    lowest_5_indices = unique_indices[sorted_indices[:5]]
    largest_5_indices = unique_indices[sorted_indices[-5:]]

    lowest_5_values = b_values[lowest_5_indices]
    largest_5_values = b_values[largest_5_indices]

    print("Lowest 5 b values and their indices:")
    for idx, val in zip(lowest_5_indices, lowest_5_values):
        print(f"Index: {idx}, Value: {val}, Angles: {angle_sets[idx]}")

    print("\nLargest 5 b values and their indices:")
    for idx, val in zip(largest_5_indices, largest_5_values):
        print(f"Index: {idx}, Value: {val}, Angles: {angle_sets[idx]}")