import xarray as xr
import numpy as np
import pandas as pd
import cftime

__all__ = ["annual_cycle"]


def annual_cycle(xobj, func, tcoord="time", time_axis_year=None):
    """Function to calculate annual cycle climatology
    This function calculates the annual cycle climatology from an
    xarray dataset containing daily timeseries variables.
    Parameters
    ----------
    xobj : xarray.core.dataset.Dataset or xarray.core.dataarray.DataArray
        Input xarray object
    tcoord : str, optional
        Name of time coordinate, by default "time"
    func : str, optional
        "mean", "std", "min", "max", or "pct90" across for the climatology,
        by default "mean"
    time_axis_year : int, optional
        Specify year used in resulting time axis, otherwise use the
        mean year for the entire dataset, by default None
    Returns
    -------
    xarray.core.dataset.Dataset
        Annual cycle climatology with 365 time points, 1 per day of year
        (leap days are dropped)
    """
    import xarray as xr
    import numpy as np
    import pandas as pd
    import cftime

    # Try to import the compiled Cython module
    try:
        import fast_percentile

        cython_available = True
        print("Using Cython-accelerated percentile calculation")
    except ImportError:
        print(
            "Warning: Cython module not available. Using slower Python implementation."
        )
        cython_available = False

    # Extract calendar and handle Dataset vs DataArray
    calendar = xobj[tcoord].values[0].calendar
    dim_coords = set(xobj.dims).union(set(xobj.coords))

    if isinstance(xobj, xr.core.dataset.Dataset):
        variables = set(xobj.variables) - dim_coords
        _xobj = xr.Dataset()
        for var in variables:
            if xobj[var].dtype not in ["object", "timedelta64[ns]"]:
                _xobj[var] = xobj[var]
    else:
        _xobj = xobj

    # Calculate middle year (same as original)
    if time_axis_year is not None:
        midyear = int(time_axis_year)
    else:
        endyr = xobj[tcoord].values[-1]
        startyr = xobj[tcoord].values[0]
        delta = (endyr - startyr) / 2
        midyear = startyr + delta
        midyear = midyear.year

    # Extract day of year information
    time_values = _xobj[tcoord].values
    dayofyear = np.array(
        [
            d.timetuple().tm_yday if hasattr(d, "timetuple") else int(d.strftime("%j"))
            for d in time_values
        ]
    )

    # Filter out leap days (day 366)
    non_leap_mask = dayofyear != 366
    if not np.all(non_leap_mask):
        # Create a new filtered dataset without leap days
        if isinstance(_xobj, xr.Dataset):
            _xobj = _xobj.isel({tcoord: non_leap_mask})
        else:
            _xobj = _xobj.isel({tcoord: non_leap_mask})
        # Update dayofyear array
        time_values = _xobj[tcoord].values
        dayofyear = np.array(
            [
                d.timetuple().tm_yday
                if hasattr(d, "timetuple")
                else int(d.strftime("%j"))
                for d in time_values
            ]
        )

    # For percentile calculation, use the fast Cython implementation if available
    if func == "pct90":
        # Check if we have the right dimensionality for the Cython version
        if isinstance(_xobj, xr.Dataset):
            result_dict = {}
            for var_name, var_data in _xobj.data_vars.items():
                # Skip non-numeric variables
                if var_data.dtype in ["object", "timedelta64[ns]"]:
                    continue

                data = var_data.values

                # Use Cython for 4D arrays if available
                if (
                    cython_available
                    and len(data.shape) == 4
                    and data.shape[0] == len(dayofyear)
                ):
                    # Ensure data is in float64 for Cython function
                    if data.dtype != np.float64:
                        data = data.astype(np.float64)

                    # Call the Cython implementation
                    result_array, unique_days = fast_percentile.fast_percentile90_4d(
                        data, dayofyear.astype(np.int32)
                    )

                    # Create DataArray with the result
                    coords = {}
                    dims = list(var_data.dims)
                    dims[0] = "dayofyear"  # Replace time dimension with dayofyear

                    # Copy other coordinates
                    for i, dim in enumerate(dims):
                        if dim != "dayofyear":
                            coords[dim] = var_data.coords[dim]

                    # Add dayofyear coordinate
                    coords["dayofyear"] = unique_days

                    # Create DataArray
                    result_dict[var_name] = xr.DataArray(
                        result_array, dims=dims, coords=coords
                    )
                else:
                    # Fall back to standard xarray implementation
                    doy_array = xr.DataArray(
                        dayofyear, dims=tcoord, coords={tcoord: _xobj[tcoord]}
                    )

                    # Add dayofyear as a coordinate
                    temp_var = var_data.assign_coords({"dayofyear": doy_array})

                    # Calculate quantile using xarray's method
                    result_dict[var_name] = temp_var.groupby("dayofyear").quantile(
                        0.90, dim=tcoord
                    )

            # Combine the results into a dataset
            result = xr.Dataset(result_dict)

        else:
            # DataArray case
            data = _xobj.values

            # Use Cython for 4D arrays if available
            if (
                cython_available
                and len(data.shape) == 4
                and data.shape[0] == len(dayofyear)
            ):
                # Ensure data is in float64 for Cython function
                if data.dtype != np.float64:
                    data = data.astype(np.float64)

                # Call the Cython implementation
                result_array, unique_days = fast_percentile.fast_percentile90_4d(
                    data, dayofyear.astype(np.int32)
                )

                # Create DataArray with the result
                coords = {}
                dims = list(_xobj.dims)
                dims[0] = "dayofyear"  # Replace time dimension with dayofyear

                # Copy other coordinates
                for i, dim in enumerate(dims):
                    if dim != "dayofyear":
                        coords[dim] = _xobj.coords[dim]

                # Add dayofyear coordinate
                coords["dayofyear"] = unique_days

                # Create DataArray
                result = xr.DataArray(result_array, dims=dims, coords=coords)
            else:
                # Fall back to standard xarray implementation
                doy_array = xr.DataArray(
                    dayofyear, dims=tcoord, coords={tcoord: _xobj[tcoord]}
                )

                # Add dayofyear as a coordinate
                temp_obj = _xobj.assign_coords({"dayofyear": doy_array})

                # Calculate quantile using xarray's method
                result = temp_obj.groupby("dayofyear").quantile(0.90, dim=tcoord)

    else:
        # For non-percentile calculations, use the standard xarray implementation
        doy_array = xr.DataArray(dayofyear, dims=tcoord, coords={tcoord: _xobj[tcoord]})

        # Create a temporary coordinate with the DOY values
        temp_obj = _xobj.assign_coords({"dayofyear": doy_array})

        # Group by the dayofyear coordinate directly
        if func == "mean":
            result = temp_obj.groupby("dayofyear").mean(tcoord)
        elif func == "min":
            result = temp_obj.groupby("dayofyear").min(tcoord)
        elif func == "max":
            result = temp_obj.groupby("dayofyear").max(tcoord)
        elif func == "std":
            result = temp_obj.groupby("dayofyear").std(tcoord)
        else:
            raise ValueError(f"Unknown argument 'func={func}' to annual cycle")

    # If there are any missing days of year in the result, fill them
    # This ensures we have all 365 days in the output (excluding leap day)
    all_days = np.arange(1, 366)  # Always use 365 days
    existing_days = result.dayofyear.values
    missing_days = set(all_days) - set(existing_days)

    if missing_days and len(missing_days) < len(
        all_days
    ):  # Only fill if we have some data
        # Get a reference day for filling
        ref_day = existing_days[0]
        for missing_day in missing_days:
            # Find closest available day
            closest_day = min(existing_days, key=lambda x: abs(x - missing_day))
            # Fill with data from closest day (simple approach)
            if (
                missing_day in all_days[: len(existing_days)]
            ):  # Ensure we're within bounds
                fill_data = result.sel(dayofyear=closest_day)
                # Add this day to the result
                result = xr.concat(
                    [result, fill_data.assign_coords(dayofyear=missing_day)],
                    dim="dayofyear",
                )

    # Sort by dayofyear to ensure proper order
    result = result.sortby("dayofyear")

    # Create proper time coordinates
    if time_axis_year is not None:
        year = int(time_axis_year)
    else:
        year = int(midyear)

    # Ensure year is properly formatted with 4 digits
    year_str = f"{year:04d}"

    # Create daily time axis for the climatology (always 365 days)
    time_values = []
    for doy in result.dayofyear.values:
        # Calculate the date for this day of year
        month, day = _day_of_year_to_month_day(doy)

        # Create the date with properly formatted year
        time_values.append(
            xr.cftime_range(
                f"{year_str}-{month:02d}-{day:02d}", periods=1, calendar=calendar
            )[0]
        )

    # Assign the time values to the result
    result = result.assign_coords({tcoord: ("dayofyear", time_values)})

    # Rename dayofyear to time coordinate
    result = result.swap_dims({"dayofyear": tcoord})

    # Drop the dayofyear coordinate
    if "dayofyear" in result.coords:
        result = result.drop_vars("dayofyear")

    return result


def _day_of_year_to_month_day(doy, leap=False):
    """Convert day of year to month and day.

    Parameters
    ----------
    doy : int
        Day of year (1-366)
    leap : bool, optional
        Whether this is a leap year, by default False

    Returns
    -------
    tuple
        (month, day) tuple
    """
    # Days in each month (non-leap year)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if leap:
        days_in_month[1] = 29  # February has 29 days in leap years

    month = 1
    day = doy

    for days in days_in_month:
        if day <= days:
            break
        day -= days
        month += 1

    return month, day
