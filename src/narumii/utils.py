import sys

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
        *files: object
        ) -> None:
        self.files = files
    
    def write(
        self, 
        obj: object
        ) -> None:
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self) -> None:
        for f in self.files:
            f.flush()