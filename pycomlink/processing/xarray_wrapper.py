from functools import wraps
from collections import OrderedDict
import inspect
import numpy as np
import xarray as xr


def xarray_loop_vars_over_dim(vars_to_loop, loop_dim):
    """
    A decorator to feed CML data with mutiple channels as xarray.DataArrays to CML processing functions

    Parameters
    ----------
    vars_to_loop: list of strings
        List of the names of the variables used as kwargs in the decorated function
        which should have a dimension `loop_dim` for which the decorated function is
        then applied individually to each item when looping over `loop_dim`.
    loop_dim: basestring
        Name of the dimension which all variables in `vars_to_loop` must have in common
        and which will be looped over to apply the decorated function.

    Examples
    --------
    Here is an example for how this decorator is used for the WAA Schleiss function::

        @xarray_loop_vars_over_dim(vars_to_loop=["rsl", "baseline", "wet"], loop_dim="channel_id")
        def waa_schleiss_2013(rsl, baseline, wet, waa_max, delta_t, tau):
            # function body...

    Here, `delta_t` and `tau` are not CML data xarray.DataArrays and hence do not
    have to be looped over.

    """
    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            # TODO: Maybe check if all args or kwargs are the same type
            # The case with numpy array as arg
            if len(args) > 0 and isinstance(args[0], np.ndarray):
                return func(*args, **kwargs)
            # The case with numpy array as kwarg
            if len(kwargs.keys()) > 0 and isinstance(
                kwargs[vars_to_loop[0]], np.ndarray
            ):
                return func(*args, **kwargs)
            # The dummy case where nothing is passed. This is just to get the
            # functions error message here instead of continuing to the loop below
            if len(args) == 0 and len(kwargs) == 0:
                return func(*args, **kwargs)

            loop_dim_id_list = list(
                np.atleast_1d(kwargs[vars_to_loop[0]][loop_dim].values)
            )
            if len(loop_dim_id_list) > 1:
                kwargs_vars_to_loop = {var: kwargs.pop(var) for var in vars_to_loop}
                data_list = []
                for loop_dim_id in loop_dim_id_list:
                    for var in vars_to_loop:
                        kwargs[var] = kwargs_vars_to_loop[var].sel(
                            {loop_dim: loop_dim_id}
                        ).values
                    data_list.append(func(**kwargs))
                return xr.DataArray(
                    data=np.stack(data_list),
                    dims=(loop_dim, "time"),
                    coords={
                        loop_dim: kwargs_vars_to_loop[vars_to_loop[0]][loop_dim].values,
                        "time": kwargs_vars_to_loop[vars_to_loop[0]].time,
                    },
                )
            else:
                return xr.DataArray(
                    data=func(**kwargs),
                    dims=("time"),
                    coords={"time": kwargs[vars_to_loop[0]].time},
                )

        return inner

    return decorator


def _get_new_args_dict(func, args, kwargs):
    """Build one dict from args, kwargs and function default args

    The function signature is used to build one joint dict from args and kwargs and
    additional from the default arguments found in the function signature. The order
    of the args in this dict is the order of the args in the function signature and
    hence the list of args can be used in cases where we can only supply *args, but
    we have to work with a mixture of args, kwargs and default args as in
    xarray.apply_ufunc in the xarray wrapper.

    """
    new_args_dict = OrderedDict()
    for i, (arg, parameter) in enumerate(inspect.signature(func).parameters.items()):
        if i < len(args):
            new_args_dict[arg] = args[i]
        elif arg in kwargs.keys():
            new_args_dict[arg] = kwargs[arg]
        else:
            new_args_dict[arg] = parameter.default

    return new_args_dict
