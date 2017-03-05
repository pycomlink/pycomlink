#----------------------------------------------------------------------------
# Name:         
# Purpose:      
#
# Authors:      
#
# Created:      
# Copyright:    (c) Christian Chwala 2016
# Licence:      The MIT License
#----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np


def xarray_pcolor(data_array,
                  ax=None,
                  mask_nan=False,
                  mask_smaller_than=None,
                  mask_larger_than=None,
                  vmin=None,
                  vmax=None,
                  cmap=plt.cm.viridis,
                  **kwargs):

    if ax is None:
        fig, ax = plt.subplots(**kwargs)

    values = data_array.values

    # Init a mask with all False values
    mask = ~np.isnan(np.zeros_like(values))

    if mask_nan:
        mask = np.isnan(values)
    if mask_smaller_than is not None:
        mask |= values < mask_smaller_than
    if mask_larger_than is not None:
        mask |= values > mask_larger_than

    values = np.ma.array(values, mask=mask)

    p_cml = ax.pcolormesh(data_array.lon,
                          data_array.lat,
                          values,
                          vmin=vmin, vmax=vmax,
                          cmap=cmap)

    return p_cml


def xarray_update_pcolor(pc,
                         data_array,
                         mask_nan=False,
                         mask_smaller_than=None,
                         mask_larger_than=None):

    values = data_array.values

    # Init a mask with all False values
    mask = ~np.isnan(np.zeros_like(values))

    if mask_nan:
        mask = np.isnan(values)
    if mask_smaller_than is not None:
        mask |= values < mask_smaller_than
    if mask_larger_than is not None:
        mask |= values > mask_larger_than

    values = np.ma.array(values, mask=mask)

    pc.set_array(values[:-1, :-1].ravel())
    pc.get_figure().canvas.draw()
