import sys
import numpy as np
from typing import TextIO


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


def read_results(
        filename: str, 
        skiprows: int = 1
        ) -> np.ndarray:
    
    results = np.loadtxt(filename, delimiter=",", skiprows=skiprows)
    results = results.transpose()

    return results


class Tee:
    '''
    Redirects output to multiple file-like objects.

    Parameters
    ----------
    *files: file-like objects
        The file-like objects to which output will be written.
    '''
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