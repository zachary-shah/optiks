import torch

def torch_interp1d(y: torch.Tensor, x: torch.Tensor, xx: torch.Tensor, check=False) -> torch.Tensor:
    """
    Performs 1D linear interpolation of PyTorch tensors allowing backpropagation
    through any of the input parameters. Interpolates from N points to M points
    over D axes. y, x, and xx must be on the same device.

    Parameters
    ----------
    y : tensor
        Dependent data to interpolate. Tensor size (N, D) or (N,).
    x : tensor
        Original independent variable values. Tensor size (N,). Must be strictly increasing.
    xx : tensor
        New independent variable values to interpolate at. Must lie within [x[0], x[-1]].
        Tensor size (M,).

    Returns
    -------
    yy : tensor
        Data y interpolated to points xx. Tensor size (M, D) if y is (N, D), else (M,).
    """

    if x.dim() != 1:
        raise ValueError(f"`x` must be 1D (got shape {tuple(x.shape)}).")
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"First dim of `y` must match len(x); got {y.shape[0]} vs {x.shape[0]}.")
    if xx.dim() != 1:
        raise ValueError(f"`xx` must be 1D (got shape {tuple(xx.shape)}).")

    # Enforce same device/dtype (torch will upcast as needed)
    device = x.device
    dtype = torch.promote_types(x.dtype, torch.promote_types(y.dtype, xx.dtype))
    x = x.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)
    xx = xx.to(device=device, dtype=dtype)

    # Basic validity checks (off by default for speed)
    if check:
        if not torch.all(x[1:] > x[:-1]):
            raise ValueError("`x` must be strictly increasing.")
        if (xx.min() < x[0]) or (xx.max() > x[-1]):
            raise ValueError("All `xx` must lie within the range of `x` (inclusive).")

    N = x.shape[0]

    # Find interval indices i such that x[i] <= xx < x[i+1]
    # (clamp to [0, N-2] to be safe on inclusivity at the right end)
    inds = torch.searchsorted(x, xx, right=False) - 1
    inds = inds.clamp(0, N - 2)

    x0 = x[inds]          # (M,)
    x1 = x[inds + 1]      # (M,)

    if y.dim() == 1:
        y0 = y[inds]          # (M,)
        y1 = y[inds + 1]      # (M,)
    else:
        y0 = y.index_select(0, inds)         # (M, D)
        y1 = y.index_select(0, inds + 1)     # (M, D)

    denom = (x1 - x0)
    if check:
        if torch.any(denom <= 0):
            raise ValueError("Found non-increasing or duplicate entries in `x`.")
    t = (xx - x0) / denom  # (M,)

    # Interpolate
    if y.dim() == 1:
        yy = y0 + t * (y1 - y0)              # (M,)
    else:
        yy = y0 + t.unsqueeze(-1) * (y1 - y0)  # (M, D)

    return yy
