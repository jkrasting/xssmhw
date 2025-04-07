import numpy as np
import xarray as xr

__all__ = ["downcast_dataset", "load_chunk", "smooth_1d", "smooth_ts"]


def downcast_dataset(ds):
    for var in ds.keys():
        ds[var] = ds[var].astype("float32")
    return ds


def load_chunk(
    ds, times=("0151-01-01", "0180-12-31"), xrange=(500, 520), yrange=(700, 720)
):
    # Load dataset
    temp_chunk = ds.thetao.sel(time=slice(*times))
    temp_chunk = temp_chunk.isel(xh=slice(*xrange), yh=slice(*yrange))
    return temp_chunk


def smooth_1d(arr_np, w=31):
    # Convert to NumPy array
    pad_width = w // 2
    padded_arr = np.pad(
        arr_np, ((pad_width, pad_width),), mode="wrap"
    )  # or 'reflect' depending on context
    # Smoothing kernel
    kernel = np.ones(w) / w
    # Apply convolution
    smoothed_np = np.convolve(padded_arr, kernel, mode="valid")
    return smoothed_np


def smooth_ts(arr):
    smoothed = np.apply_along_axis(smooth_1d, 0, arr)
    smoothed_data = xr.DataArray(smoothed, coords=arr.coords, dims=arr.dims)
    return smoothed_data
