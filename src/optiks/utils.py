import torch
import numpy as np
import subprocess
from io import BytesIO
import pandas as pd
from os import environ


def spiralTraj(fov, res, nshots=1, npoints=int(7e3)):
    """
    Designs a spiral trajectory in arbitrary parameterization with the desired k-space coverage.

    Parameters
    ----------
    fov : float
        Field of view for trajectory [cm].
    res : float
        Resolution for trajectory [cm].
    nshots : int, optional
        Number of shots to use. Defaults to 1.
    npoints : int, optional
        Number of points to discretize trajectory into. Longer trajectories require more points. Defaults to 7000.

    Returns
    -------
    C : ndarray
        K-space spiral trajectory as a 1D complex array [cm^-1].

    Notes
    -----
    (c) Matthew A. McCready 2024
    """
    nt = fov / (2 * res)
    kmax = 1 / (2 * res)
    dt = 1 / npoints
    t = np.arange(0, 1 + dt, dt)
    C = kmax * t * np.exp(1j * 2 * np.pi * nt * t)
    C = C[:, np.newaxis]
    return C


def rosetteTraj(res, n1=7, n2=5, npoints=int(1e3)):
    """
    Designs a rosette trajectory in arbitrary parameterization with the desired k-space coverage.

    Parameters
    ----------
    fov : float
        Field of view for trajectory [cm].
    res : float
        Resolution for trajectory [cm].
    nshots : int, optional
        Number of shots to use. Defaults to 1.
    npoints : int, optional
        Number of points to discretize trajectory into. Longer trajectories require more points. Defaults to 7000.

    Returns
    -------
    C : ndarray
        K-space spiral trajectory as a 1D complex array [cm^-1].

    Notes
    -----
    (c) Matthew A. McCready 2024
    """
    t = np.linspace(0, 0.5, num=npoints)
    kmax = 1 / (2 * res)
    C = kmax * np.sin(2 * np.pi * n1 * t) * np.exp(1j * 2 * np.pi * n2 * t)
    C = C[:, np.newaxis]
    return C


def get_free_gpu():
    """
    Determines index of GPU with the largest available memory and returns.
    usage: idx = get_free_gpu()

    Returns
    -------
    idx : int
        Index of GPU with the largest available memory.

    Notes
    -----
    (c) Matthew A. McCready 2024
    """
    environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # make GPU ID match that of nvitop
    print('Selecting GPU with largest free memory...')
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(BytesIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    idx = gpu_df['memory.free'].idxmax()
    print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx
