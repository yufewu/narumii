import numpy as np


# estimate error from parabolic fitting of residuals
def error_parabolic_rmsd(
    a_rmsd: float, 
    rmsd_min: float, 
    n: int, 
    p: int = 1, 
    sd_from_rmsd: bool = True
    ) -> float:
    """
    Error analysis based on parabolic fitting of root mean square deviation (RMSD)

    Parameters
    -----
    a_rmsd: float
        Curvature of the RMSD-parameter parabolic fit.
    rmsd_min: float
        Minimum RMSD value at best fit.
    n: int
        Number of data points.
    p: int, optional
        Number of fitted parameters (default is 1).
    sd_from_rmsd: bool, optional
        Whether to estimate standard deviation from RMSD (default is True).
    
    Returns
    ------
    ci: float
        Estimated half length of confidence interval for the fitted parameter.
    """
    v = n - p  # degrees of freedom
    
    if sd_from_rmsd is True:
        sd_y = np.sqrt(n/v) * rmsd_min  # Estimate standard deviation of data point from residue rmsd
    else:
        raise NotImplementedError("Standard deviation estimation from other sources is not implemented.")
    print(f"Estimated standard deviation = {sd_y:.4f}")
    
    ci = sd_y * np.sqrt(1 / (2*n*rmsd_min*a_rmsd))  # Calculate confidence interval
    print(f"Estimated confidence interval = ±{ci:.4f}")
    
    return ci


def error_parabolic_ssd(
    a_ssd: float, 
    rmsd_min: float, 
    n: int, 
    p: int = 1, 
    sd_from_rmsd: bool = True
    ) -> float:
    """
    Error analysis based on parabolic fitting of sum squared deviation (SSD)

    Parameters
    -----
    a_ssd: float
        Curvature of the SSD-parameter parabolic fit.
    rmsd_min: float
        Minimum RMSD value at best fit.
    n: int
        Number of data points.
    p: int, optional
        Number of fitted parameters (default is 1).
    sd_from_rmsd: bool, optional
        Whether to estimate standard deviation from RMSD (default is True).
    
    Returns
    ------
    ci: float
        Estimated half length of confidence interval for the fitted parameter.
    """
    
    v = n - p  # degrees of freedom
    
    if sd_from_rmsd is True:
        sd_y = np.sqrt(n/v) * rmsd_min  # Estimate standard deviation of data point from residue rmsd
    else:
        raise NotImplementedError("Standard deviation estimation from other sources is not implemented.")
    print(f"Estimated standard deviation = {sd_y:.4f}")
    
    ci = sd_y * np.sqrt(1 / (a_ssd))  # Calculate confidence interval
    print(f"Estimated confidence interval = ±{ci:.4f}")
    
    return ci