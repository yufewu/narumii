import sys
import numpy as np
from typing import TextIO


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