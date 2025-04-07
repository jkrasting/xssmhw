# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language=c++

import numpy as np
cimport numpy as np
from libc.math cimport isnan
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort as cpp_sort
from libcpp.algorithm cimport nth_element as cpp_nth_element

# Fast 90th percentile calculation for 4D arrays (time, z, y, x)
def fast_percentile90_4d(np.ndarray[double, ndim=4] data, np.ndarray[int, ndim=1] day_indices):
    """
    Calculate 90th percentile for each unique day of year in a 4D array.
    
    Parameters:
    -----------
    data : 4D numpy array (time, z, y, x)
        Input data array
    day_indices : 1D numpy array
        Day of year (1-365) for each time point
    
    Returns:
    --------
    result : 4D numpy array (days, z, y, x)
        90th percentile values for each day of year
    """
    cdef int t_size = data.shape[0]
    cdef int z_size = data.shape[1]
    cdef int y_size = data.shape[2]
    cdef int x_size = data.shape[3]
    
    # Find unique days
    cdef vector[int] unique_days
    cdef int i, j, k, l, t, day, idx
    cdef int max_day = 0
    
    # Find the maximum day value to determine array size
    for i in range(t_size):
        if day_indices[i] > max_day:
            max_day = day_indices[i]
    
    # Create lookup for days
    cdef vector[vector[int]] day_to_indices
    day_to_indices.resize(max_day + 1)  # +1 because days are 1-indexed
    
    # Collect indices for each day
    for i in range(t_size):
        day = day_indices[i]
        day_to_indices[day].push_back(i)
    
    # Find unique days with data
    for day in range(1, max_day + 1):
        if day_to_indices[day].size() > 0:
            unique_days.push_back(day)
    
    # Allocate result array
    cdef int n_days = unique_days.size()
    cdef np.ndarray[double, ndim=4] result = np.zeros((n_days, z_size, y_size, x_size), dtype=np.float64)
    
    # For each unique day, compute 90th percentile
    cdef vector[double] values
    cdef double p90
    cdef int n_values, p90_idx
    
    for day_idx in range(n_days):
        day = unique_days[day_idx]
        indices = day_to_indices[day]
        n_indices = indices.size()
        
        # For each spatial point
        for z in range(z_size):
            for y in range(y_size):
                for x in range(x_size):
                    # Collect values for this spatial point
                    values.clear()
                    for i in range(n_indices):
                        t = indices[i]
                        if not isnan(data[t, z, y, x]):
                            values.push_back(data[t, z, y, x])
                    
                    # Calculate 90th percentile
                    n_values = values.size()
                    if n_values > 0:
                        # Use C++ algorithm to sort values
                        cpp_sort(values.begin(), values.end())
                        
                        # Find 90th percentile
                        p90_idx = int(0.9 * n_values)
                        if p90_idx >= n_values:
                            p90_idx = n_values - 1
                        
                        # For exact 90th percentile, use interpolation if needed
                        if 0.9 * n_values == p90_idx:
                            p90 = values[p90_idx]
                        else:
                            # Linear interpolation
                            if p90_idx + 1 < n_values:
                                p90 = values[p90_idx] + (0.9 * n_values - p90_idx) * (values[p90_idx+1] - values[p90_idx])
                            else:
                                p90 = values[p90_idx]
                        
                        result[day_idx, z, y, x] = p90
                    else:
                        result[day_idx, z, y, x] = np.nan
    
    return result, unique_days
