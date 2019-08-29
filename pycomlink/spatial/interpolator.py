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
        """ Perform interpolation
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

        assert(len(x) == len(y) == len(z))

        z = z.astype('float')

        self.xgrid, self.ygrid = _parse_grid_kwargs(x_list=x,
                                                    y_list=y,
                                                    xgrid=xgrid,
                                                    ygrid=ygrid,
                                                    resolution=resolution)
        if (~pd.isnull(z)).sum() == 0:
            zgrid = np.zeros_like(self.xgrid)
            zgrid[:] = np.nan
            self.zgrid = zgrid

        else:
            zi = self._interpol_func(x=x,
                                     y=y,
                                     z=z,
                                     xi=self.xgrid.ravel(),
                                     yi=self.ygrid.ravel())

            self.x, self.y, self.z = x, y, z
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
            # print 'Reusing old `Invdisttree`'
            idw = self.idw
        else:
            idw = Invdisttree(X=list(zip(x, y)))
            self.idw = idw
            self.x = x
            self.y = y

        zi = idw(q=list(zip(xi, yi)),
                 z=z,
                 nnear=self.nnear,
                 p=self.p,
                 max_distance=self.max_distance)
        return zi


class OrdinaryKrigingInterpolator(PointsToGridInterpolator):
    def __init__(self,
                 nlags=100,
                 variogram_model='spherical',
                 weight=True,
                 n_closest_points=None,
                 # coordinates_type='euclidean', # Not supported in v1.3.1
                 backend='C'):
        """ A ordinary kriging interpolator for points to grid"""

        self.nlags = nlags
        self.variogram_model = variogram_model
        self.weight = weight
        self.n_closest_points = n_closest_points
        # self.coordinates_type = coordinates_type
        self.backend = backend

    def _interpol_func(self, x, y, z, xi, yi):
        ok = OrdinaryKriging(x,
                             y,
                             z,
                             nlags=self.nlags,
                             variogram_model=self.variogram_model,
                             weight=self.weight)
                             # coordinates_type=self.coordinates_type)

        zi, sigma = ok.execute(style='points',
                               xpoints=xi,
                               ypoints=yi,
                               n_closest_points=self.n_closest_points,
                               backend=self.backend)

        self.sigma = sigma
        return zi


class ComlinkGridInterpolator(object):
    """ Convenience class for interpolating CML data to grid """

    def __init__(self,
                 cml_list,
                 xgrid=None,
                 ygrid=None,
                 resolution=None,
                 interpolator=IdwKdtreeInterpolator(),
                 resample_to='H',
                 floor_before_resample_to_new_index='min',
                 resample_to_new_index=None,
                 resample_label='right',
                 variable='R',
                 channels=['channel_1'],
                 aggregation_func=np.mean,
                 apply_factor=1):

        self.lons, self.lats = get_lon_lat_list_from_cml_list(cml_list)
        # Later some coordinate transformations can be added here
        self.x = self.lons
        self.y = self.lats
        self.variable = variable

        self.df_cmls = get_dataframe_for_cml_variable(
            cml_list,
            resample_to=resample_to,
            resample_to_new_index=resample_to_new_index,
            floor_before_resample_to_new_index=
            floor_before_resample_to_new_index,
            resample_label=resample_label,
            variable=variable,
            channels=channels,
            aggregation_func=aggregation_func,
            apply_factor=apply_factor)

        self._interpolator = interpolator
        self.resolution = resolution

        self.xgrid, self.ygrid = _parse_grid_kwargs(x_list=self.x,
                                                    y_list=self.y,
                                                    xgrid=xgrid,
                                                    ygrid=ygrid,
                                                    resolution=resolution)
        self.ds_gridded = None

    def interpolate_for_i(self, i):
        """ Interpolate CML data for one specific time index
        Parameters
        ----------
        i : int
            Integer refering to time index in `DataFrame` of the aggregated CML
            data
        Returns
        -------
        zgrid : numpy.array
            Array of interpolated field
        """
        z = self.df_cmls.iloc[i, :]

        zgrid = self._interpolator(x=self.x,
                                   y=self.y,
                                   z=z,
                                   xgrid=self.xgrid,
                                   ygrid=self.ygrid)
        return zgrid

    def loop_over_time(self, t_start=None, t_stop=None):
        """ Do interpolation for many time steps
        Note: This function also updates the attribute `ds_gridded`.
        Parameters
        ----------
        t_start : str, optional
            Starting time for interpolation loop
        t_stop : str, optional
            Stop time for interpolation loop
        Returns
        -------
        ds_gridded : xarray.Dataset
            Dataset of the gridded fields, including x- and y-grid and
            timestamps
        """
        zi_list = []

        for i in tqdm(list(range(len(self.df_cmls[t_start:t_stop].index)))):
            try:
                zi = self.interpolate_for_i(i)
            except (scipy.linalg.LinAlgError, ValueError) as e:
                # Catch Kriging error and return NaNs
                if e.args[0].lower() == 'singular matrix':
                    print('%s: Kriging calculations produced '
                          'singular matrix. Returning NaNs.'
                          % self.df_cmls.index[i])
                    zi = np.ones_like(self._interpolator.xgrid)
                    zi[:] = np.nan
                    sigma = np.ones_like(self._interpolator.xgrid)
                    sigma[:] = np.nan
                else:
                    print('baz')
                    raise e
            zi_list.append(zi)

        self.ds_gridded = self._fields_to_dataset(field_list=zi_list,
                                                  t_start=t_start,
                                                  t_stop=t_stop)
        return self.ds_gridded

    def _fields_to_dataset(self, field_list, t_start=None, t_stop=None):
        if t_start is None:
            t_start = self.df_cmls.index[0]
        if t_stop is None:
            t_stop = self.df_cmls.index[-1]

        ds = xr.Dataset(
            data_vars={self.variable: ((['time', 'y', 'x'],
                                        np.array(field_list)))},
            coords={'lon': (['y', 'x'], self.xgrid),
                    'lat': (['y', 'x'], self.ygrid),
                    'time': (self.df_cmls[t_start:t_stop]
                             .index.values
                             .astype(np.datetime64))})
        return ds


def _parse_grid_kwargs(x_list, y_list, xgrid, ygrid, resolution):
    """ Generate grids if None is supplied
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
            raise ValueError('`resolution must be set if `xgrid` '
                             'or `ygrid` are None')

        xcoords = np.arange(min(x_list) - resolution,
                            max(x_list) + resolution,
                            resolution)
        ycoords = np.arange(min(y_list) - resolution,
                            max(y_list) + resolution,
                            resolution)
        xgrid, ygrid = np.meshgrid(xcoords, ycoords)
    else:
        pass
    return xgrid, ygrid


def get_lon_lat_list_from_cml_list(cml_list):
    """ Extract lats and lons from all CMLs """

    lons = np.array(
        [cml.get_center_lon_lat()[0] for cml in cml_list])
    lats = np.array(
        [cml.get_center_lon_lat()[1] for cml in cml_list])
    return lons, lats


def get_dataframe_for_cml_variable(cml_list,
                                   resample_to='H',
                                   resample_to_new_index=None,
                                   resample_label='right',
                                   floor_before_resample_to_new_index='min',
                                   variable='R',
                                   channels=['channel_1'],
                                   aggregation_func=np.mean,
                                   apply_factor=1):
    """ Build a DataFrame for a certain variable for all CMLs
    The column names of the resulting `DataFrame` are the `cml_id`s of
    each CML. The temporal aggregation of the CML data contained in the
    `ComlinkChannel.data` `DataFrame`.
    Parameters
    ----------
    cml_list : iterable of `Comlink` objects
    resample_to : str, optional
        `pandas` resampling string, defaults to 'H
    resample_to_new_index : iterable of datetime or something similar, optional
        Time stamps of a new index on which to aggregate. If this argument is
        supplied, `resample_to` is ignored and the CML data is aggregated to
        this new index, which can have arbitrary time steps.
    floor_before_resample_to_new_index : str
        Per default this is set to 'min' and will floor the CML DataFrame
        DatetimeIndex to minutes before doing a custom aggregation to a new
        index.
    resample_label : {'left', 'right'}, optional
        `pandas` resampling label, defaults to 'right'
    variable : str, optional
        Column name used in the `DataFrame` in `ComlinkChannel.data` for
        the data that shall be aggregated
    channels : interable, optional
        List of the channel names to use for
    aggregation_func
    apply_factor
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with one column for each CML
    """

    # TODO: Extend the code to be able to average over two channels if desired
    channel_name = channels[0]

    if resample_to_new_index is not None:
        df_dict = {}
        for cml in cml_list:
            df_cml = cml.channels[channel_name].data[variable]
            if floor_before_resample_to_new_index:
                df_cml.index = (
                    df_cml.index.floor(floor_before_resample_to_new_index))
            df_dict[cml.metadata['cml_id']] = df_cml
        df = pd.concat(df_dict, axis=1)
        df = aggregate_df_onto_DatetimeIndex(
            df=df,
            new_index=resample_to_new_index,
            label=resample_label,
            method=aggregation_func)

    else:
        df_dict = {}
        for cml in cml_list:
            df_dict[cml.metadata['cml_id']] = (
                cml.channels[channel_name].data[variable]
                .resample(resample_to, label=resample_label)
                .apply(aggregation_func))
        df = pd.concat(df_dict, axis=1)

    # Assure the correct order of the columns.
    # Please note that the order of the columns might get mixed up even if a
    # OrderdDictionary of DataFrames is used for concatenation. Reordering
    # the columns is computationally cheap compare to the rest of the
    # interpolation.
    cml_id_list = [cml.metadata['cml_id'] for cml in cml_list]
    df = df[cml_id_list]

    df *= apply_factor

    return df


