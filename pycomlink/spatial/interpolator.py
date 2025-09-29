from __future__ import print_function
from builtins import zip
from builtins import range
from builtins import object
import abc
import numpy as np
import scipy
import pandas as pd
import xarray as xr
from tqdm import tqdm
from pykrige import OrdinaryKriging
import math
pi = math.pi
from datetime import datetime

from .idw import Invdisttree
from ..util.temporal import aggregate_df_onto_DatetimeIndex

from future.utils import with_metaclass


class PointsToGridInterpolator(with_metaclass(abc.ABCMeta, object)):
    """PointsToGridInterpolator class docstring"""

    @abc.abstractmethod
    def __init__(self):
        """Pass all configuration parameters for the interpolator here"""
        return

    def __call__(self, x, y, z, xgrid=None, ygrid=None, resolution=None):
        """Perform interpolation
        This calls the actual internal interpolation function. Passing `x` and
        `y` every time is not optimal for performance, but subclasses might
        be implemented to reuse precalculated information of `x` and `y` have
        not change comapred to last call to interpolation function.
        Parameters
        ----------
        x : array-like
        y : array-like
        z : array-like
        xgrid : 2D array
        ygrid : 2D array
        resolution : float
        Returns
        -------
        zgrid : interpolated data with shape of `xgrid` and `ygrid`
        """

        assert len(x) == len(y) == len(z)

        z = z.astype("float")

        self.xgrid, self.ygrid = _parse_grid_kwargs(
            x_list=x, y_list=y, xgrid=xgrid, ygrid=ygrid, resolution=resolution
        )
        if (~pd.isnull(z)).sum() == 0:
            zgrid = np.zeros_like(self.xgrid)
            zgrid[:] = np.nan
            self.zgrid = zgrid

        else:
            zi = self._interpol_func(
                x=x, y=y, z=z, xi=self.xgrid.ravel(), yi=self.ygrid.ravel()
            )
            self.zgrid = np.reshape(zi, self.xgrid.shape)

        return self.zgrid

    @abc.abstractmethod
    def _interpol_func(self, x, y, z, xi, yi):
        """The actual interpolation code goes here"""
        return


class IdwKdtreeInterpolator(PointsToGridInterpolator):
    def __init__(self, nnear=8, p=2, exclude_nan=True, max_distance=None):
        """A k-d tree based IDW interpolator for points to grid"""
        self.nnear = nnear
        self.p = p
        self.exclude_nan = exclude_nan
        self.max_distance = max_distance
        self.x = None
        self.y = None

    def _interpol_func(self, x, y, z, xi, yi):
        """Do IDW interpolation"""

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        if self.exclude_nan:
            not_nan_ix = ~np.isnan(z)
            x = x[not_nan_ix]
            y = y[not_nan_ix]
            z = z[not_nan_ix]
        self.z = z

        if np.array_equal(x, self.x) and np.array_equal(y, self.y):
            # print('Reusing old `Invdisttree`')
            idw = self.idw
        else:
            # print('Building new `Invdistree`')
            idw = Invdisttree(X=list(zip(x, y)))
            self.idw = idw
            self.x = x
            self.y = y

        zi = idw(
            q=list(zip(xi, yi)),
            z=z,
            nnear=self.nnear,
            p=self.p,
            max_distance=self.max_distance,
        )
        return zi


class OrdinaryKrigingInterpolator(PointsToGridInterpolator):
    def __init__(
        self,
        nlags=100,
        variogram_model="spherical",
        variogram_parameters={"sill": 0.9, "range": 1, "nugget": 0.1},
        weight=True,
        n_closest_points=None,
        exclude_nan=True,
        # coordinates_type='euclidean', # Not supported in v1.3.1
        backend="C",
    ):
        """A ordinary kriging interpolator for points to grid"""

        self.nlags = nlags
        self.variogram_model = variogram_model
        self.variogram_parameters = variogram_parameters
        self.weight = weight
        self.n_closest_points = n_closest_points
        self.exclude_nan = exclude_nan
        # self.coordinates_type = coordinates_type
        self.backend = backend

    def _interpol_func(self, x, y, z, xi, yi):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        if self.exclude_nan:
            not_nan_ix = ~np.isnan(z)
            x = x[not_nan_ix]
            y = y[not_nan_ix]
            z = z[not_nan_ix]
        self.z = z

        ok = OrdinaryKriging(
            x,
            y,
            z,
            nlags=self.nlags,
            variogram_model=self.variogram_model,
            variogram_parameters=self.variogram_parameters,
            weight=self.weight,
        )
        # coordinates_type=self.coordinates_type)

        zi, sigma = ok.execute(
            style="points",
            xpoints=xi,
            ypoints=yi,
            n_closest_points=self.n_closest_points,
            backend=self.backend,
        )

        self.sigma = sigma
        return zi


def _parse_grid_kwargs(x_list, y_list, xgrid, ygrid, resolution):
    """Generate grids if None is supplied
    If `xgrid` and `ygrid` are None, a grid with a spatial resolution of
    `resolution` is generated using the bounding box defined by the minima
    and maxima of `x_list` and `y_list`.
    Parameters
    ----------
    x_list
    y_list
    xgrid
    ygrid
    resolution
    Returns
    -------
    """

    if (xgrid is None) or (ygrid is None):
        if resolution is None:
            raise ValueError(
                "`resolution must be set if `xgrid` " "or `ygrid` are None"
            )

        xcoords = np.arange(
            min(x_list) - resolution, max(x_list) + resolution, resolution
        )
        ycoords = np.arange(
            min(y_list) - resolution, max(y_list) + resolution, resolution
        )
        xgrid, ygrid = np.meshgrid(xcoords, ycoords)
    else:
        pass
    return xgrid, ygrid


def clim_var_param(date_str="20130404", time_scale_hours=0.25):
    """ Obtain climatological values of sill, range, and nugget of spherical variogram model
    This is based on a climatological variogram based on 30-year automatic rain gauge data sets from The Netherlands. 
    Spherical variograms have been modelled as function of the day number and duration in Van de Beek et al. (2012). 
    They use durations of 1 - 24 h. In this function the relationships can be extrapolated to, e.g. 15-min, data.
    Returns the values of sill, range and nugget. The nugget is set to 0.1 * sill.
    Python implementation of the function "ClimVarParam.R" from the R RAINLINK package: Retrieval algorithm for rainfall mapping from microwave links
    in a cellular communication network (Overeem et al., 2016).

    References:
    Overeem, A., Leijnse, H., and Uijlenhoet, R., 2016: Retrieval algorithm for rainfall mapping from microwave links in a 
    cellular communication network, Atmospheric Measurement Techniques, 9, 2425-2444, https://doi.org/10.5194/amt-9-2425-2016.
    Van de Beek, C. Z., Leijnse, H., Torfs, P. J. J. F., and Uijlenhoet, R., 2012: Seasonal semi-variance of Dutch 
    rainfall at hourly to daily scales, Adv. Water Resour., 45, 76-85, doi:10.1016/j.advwatres.2012.03.023.

    Parameters
    ----------
    date_str: the end date of the chosen daily period in a format that pandas.to_datetime can parse e.g. 'YYYYMMDD'
    time_scale_hours: rainfall aggregation interval in hours
    
    Returns
    -------
    dict with 'sill', 'range' and 'nugget' keys.

    """

    # Set frequency (1 day expressed in years):
    frequency_years = 1/365

    # Determine day of year (Julian day number):
    date = pd.to_datetime(date_str)
    #date = datetime(int(date_str[0:4]),int(date_str[4:6]),int(date_str[6:8]))
    julian_day = float(date.strftime('%j'))

    # Calculate sill, range and nugget of spherical variogram for this particular day:
    range_m = (
        15.51 * time_scale_hours**0.09 
        + 2.06 * time_scale_hours**-0.12 
        * np.cos(2*pi*frequency_years * (julian_day - 7.37 * time_scale_hours**0.22)) 
    ) ** 4
    sill = (
        0.84 * time_scale_hours**-0.25 
        + 0.20 * time_scale_hours**-0.37 
        * np.cos(2*pi*frequency_years * (julian_day - 162 * time_scale_hours**-0.03))
    ) ** 4
	
    nugget = 0.1 * sill 
    range_km = range_m/1000 # range (in kilometers)

    return {'sill': sill, 'range': range_km, 'nugget': nugget}
