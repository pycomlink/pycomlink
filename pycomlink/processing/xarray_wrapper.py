from functools import wraps
from collections import OrderedDict
import inspect
import xarray as xr


def xarray_apply_along_time_dim():
    """
    A decorator to apply CML processing function along the time dimension of DataArrays

    This will work if the decorated function takes 1D numpy arrays, which hold the
    CML time series data, as arguments. Additional argument are also handled.

    """

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            new_args_dict = _get_new_args_dict(func=func, args=args, kwargs=kwargs)

            # Check if any arg has a time dim. Also build the input_core_dims
            # list which will be required below and in our case will contain either
            # ['time'] or [] because we do only care about applying along the time
            # dim if the arg has one. Note that input_core_dims has to have the same
            # number of entries as func has args.
            input_core_dims = []
            found_time_dim = False
            for arg in new_args_dict.values():
                try:
                    if "time" in arg.dims:
                        found_time_dim = True
                        input_core_dims.append(["time"])
                    else:
                        input_core_dims.append([])
                except AttributeError:
                    input_core_dims.append([])

            # If now arg has a `time` dim, then we just call the function as it would
            # be called without the wrapper.
            if not found_time_dim:
                return func(*args, **kwargs)
            else:
                return xr.apply_ufunc(
                    func,
                    *list(new_args_dict.values()),
                    input_core_dims=input_core_dims,
                    output_core_dims=[["time"]],
                    vectorize=True,
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
