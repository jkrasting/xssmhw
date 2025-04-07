import xarray as xr
import numpy as np

from util import downcast_dataset

__all__ = [
    "ann_duration",
    "ann_intensity",
    "ann_peak_intensity",
    "ann_freq",
    "calc_intensity" "calc_metrics",
    "cumulative_heatstress",
    "hw_days_per_year",
]


def ann_duration(intensity):
    return intensity.groupby("time.year".count("time")).mean("year")


def ann_intensity(intensity):
    mean_intensity = intensity.groupby("time.year").mean("time")
    mean_intensity = mean_intensity.fillna(0.0).mean("year")
    return mean_intensity


def ann_peak_intensity(intensity):
    peak_intensity = intensity.groupby("time.year").max("time")
    peak_intensity = peak_intensity.fillna(0.0).mean("year")
    return peak_intensity


def ann_freq(event_index):
    return event_index.max("time") / len(set(list(event_index.time.dt.year.values)))


def calc_intensity(temp_chunk, thresh_tiled, event_index):
    return xr.where(event_index > 0, temp_chunk - thresh_tiled, np.nan)


def calc_metrics(event_index, temp_chunk, thresh_tiled):
    dsout = xr.Dataset()
    dsout["clim_freq"] = ann_freq(event_index)
    intensity = calc_intensity(temp_chunk, thresh_tiled, event_index)
    dsout["clim_dur"] = hw_days_per_year(intensity)
    dsout["clim_Im"] = ann_intensity(intensity)
    dsout["clim_Atot"] = cumulative_heatstress(intensity)
    dsout["clim_HSpeak"] = ann_peak_intensity(intensity)
    dsout = downcast_dataset(dsout)
    return dsout


def cumulative_heatstress(intensity):
    cumuluative_hs = intensity.groupby("time.year").sum("time")
    cumuluative_hs = cumuluative_hs.fillna(0.0).mean("year")
    return cumuluative_hs


def hw_days_per_year(intensity):
    return intensity.groupby("time.year").count("time").mean("year")
