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

from .idw import Invdisttree
from ..util.temporal import aggregate_df_onto_DatetimeIndex

from future.utils import with_metaclass


class PointsToGridInterpolator(with_metaclass(abc.ABCMeta, object)):
    """ PointsToGridInterpolator class docstring """

    @abc.abstractmethod
    def __init__(self):
        """ Pass all configuration parameters for the interpolator here """
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
        """ The actual interpolation code goes here """
        return


class IdwKdtreeInterpolator(PointsToGridInterpolator):
    def __init__(self, nnear=8, p=2, exclude_nan=True, max_distance=None):
        """ A k-d tree based IDW interpolator for points to grid """
        self.nnear = nnear
        self.p = p
        self.exclude_nan = exclude_nan
        self.max_distance = max_distance
        self.x = None
        self.y = None

    def _interpol_func(self, x, y, z, xi, yi):
        """ Do IDW interpolation """

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
            #print('Reusing old `Invdisttree`')
            idw = self.idw
        else:
            #print('Building new `Invdistree`')
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
        weight=True,
        n_closest_points=None,
        # coordinates_type='euclidean', # Not supported in v1.3.1
        backend="C",
    ):
        """ A ordinary kriging interpolator for points to grid"""

        self.nlags = nlags
        self.variogram_model = variogram_model
        self.weight = weight
        self.n_closest_points = n_closest_points
        # self.coordinates_type = coordinates_type
        self.backend = backend

    def _interpol_func(self, x, y, z, xi, yi):
        ok = OrdinaryKriging(
            x,
            y,
            z,
            nlags=self.nlags,
            variogram_model=self.variogram_model,
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


def get_lon_lat_list_from_cml_list(cml_list):
    """ Extract lats and lons from all CMLs """

    lons = np.array([cml.get_center_lon_lat()[0] for cml in cml_list])
    lats = np.array([cml.get_center_lon_lat()[1] for cml in cml_list])
    return lons, lats
