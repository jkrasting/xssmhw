import xarray as xr
import numpy as np
import scipy.ndimage as ndimage

from annualcycle import annual_cycle
from util import smooth_ts

__all__ = [
    "calcualte_exceedences",
    "tag_events_1d",
    "tag_events",
    "threshold_and_climo",
]


def calculate_execeedences(temp_chunk, thresh):
    nyears = int(len(temp_chunk.time) / len(thresh.time))
    thresh_tiled = xr.concat([thresh] * nyears, "time")
    thresh_tiled = thresh_tiled.assign_coords({"time": temp_chunk.time})
    exceedences = temp_chunk > thresh_tiled
    return (exceedences, thresh_tiled)


def tag_events_1d(arr, min_duration=5):
    events, num_events = ndimage.label(arr)
    feature_sizes = ndimage.sum(arr, events, range(1, num_events + 1))
    mask = np.isin(events, np.where(feature_sizes >= min_duration)[0] + 1)
    events = np.where(mask, events, 0.0)
    # extract the filtered event_number
    num_events_controled = events[events != 0]
    # Define the maximum distance between events to be merged
    max_distance = 2
    num_events_controled = np.unique(events[events != 0])
    num_events_controled = num_events_controled.astype(int)
    back_order_events = np.arange(1, len(num_events_controled))[::-1]
    # Iterate through the events and merge them if the distance is within the threshold
    for n in back_order_events:
        # event number
        current_event = num_events_controled[n]
        previous_event = num_events_controled[n - 1]
        # indexs of corresponding event
        current_event_indices = np.where(events == current_event)
        previous_event_indices = np.where(events == previous_event)

        if len(current_event_indices[0]) == 0 or len(previous_event_indices[0]) == 0:
            continue

        distance_between_events = (
            current_event_indices[0][0] - previous_event_indices[0][-1] - 1
        )

        # assign the same event number for the gaps
        if distance_between_events <= max_distance:
            current_event_end = current_event_indices[0][-1]
            previous_event_start = previous_event_indices[0][0]
            events[previous_event_start : current_event_end + 1] = previous_event
    final_labeled_vector, num_final_events = ndimage.label(events > 0)
    return final_labeled_vector


def tag_events(arr):
    event_index = np.apply_along_axis(tag_events_1d, 0, arr)
    num_events = np.max(event_index, axis=0)

    event_index = xr.DataArray(event_index, dims=arr.dims, coords=arr.coords)
    num_events = xr.DataArray(
        num_events,
        dims=arr.isel(time=0).squeeze().dims,
        coords=arr.isel(time=0).squeeze().coords,
    )
    return (event_index, num_events)


def threshold_and_climo(temp_chunk, smooth=True):
    clim = annual_cycle(temp_chunk, func="mean")
    thresh = annual_cycle(temp_chunk, func="pct90")

    clim = smooth_ts(clim)
    thresh = smooth_ts(thresh)

    return (clim, thresh)
