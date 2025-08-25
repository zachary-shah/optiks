from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
import torch
from optiks.interp import torch_interp1d
from optiks.old.utils import tensorInterp

"""
Compare old and new interpolation methods against numpy
"""

N = 1000 # num trials
device_idx = 4
T = 10000 # length of signal generated
M = 10 # upsample factor for interpolation
device= torch.device(device_idx)

times = dict(
    torch_interp1d=[],
    tensorInterp=[]
)
errs = dict(
    torch_interp1d=[],
    tensorInterp=[]
)

pbar = tqdm(total=N, leave=True)
for _ in range(N):

    # generate signal
    x = torch.linspace(0, 2 * torch.pi, T)
    y = torch.sin(x) + 0.1 * torch.randn(T)  # add some noise
    x1 = torch.linspace(0.1, 2 * torch.pi - 0.1, T * M)
    y1 = torch.sin(x1)

    # to device
    x = x.to(device)
    y = y.to(device)
    x1 = x1.to(device)
    y1 = y1.to(device)

    # true interp with numpy
    true = interp1d(
        x.detach().cpu().numpy(), y.detach().cpu().numpy(), kind='linear', fill_value="extrapolate"
    )(x1.detach().cpu().numpy())

    # run interpolation methods and time them
    for method in times.keys():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        if method == "torch_interp1d":
            pred = torch_interp1d(y, x, x1)
        elif method == "tensorInterp":
            pred = tensorInterp(y, x, x1)
        end.record()
        end.synchronize()
        times[method].append(start.elapsed_time(end))

        pred = pred.detach().cpu().numpy()

        errs[method].append(np.mean((true - pred) ** 2))
    pbar.update(1)

pbar.close()

# summarize
times = {method: np.mean(times[method]) for method in times}
errs = {method: np.mean(errs[method]) for method in errs}

for method in times:
    print(f"{method}: time = {times[method]:.2f} ms, error = {errs[method]:.2e}")
