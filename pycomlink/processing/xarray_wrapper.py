from functools import wraps
import numpy as np
import xarray as xr


def xarray_loop_vars_over_dim(vars_to_loop, loop_dim):
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
                        )
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
