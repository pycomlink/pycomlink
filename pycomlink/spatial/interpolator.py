import abc
import numpy as np
import pandas as pd
from tqdm import tqdm
from pykrige import OrdinaryKriging

from .idw import Invdisttree


class PointsToGridInterpolator(object):
    """ PointsToGridInterpolator class docstring """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """ Pass all configuration parameters for the interpolator here """
        return

    def __call__(self, x, y, z, xgrid=None, ygrid=None, resolution=None):
        """ Perform the interpolation """
        assert(len(x) == len(y) == len(z))

        self.xgrid, self.ygrid = _generate_grid(x=x,
                                                y=y,
                                                xgrid=xgrid,
                                                ygrid=ygrid,
                                                resolution=resolution)

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
    def __init__(self, nnear=8, p=2, exclude_nan=True):
        """ A k-d tree based IDW interpolator for points to grid """
        self.nnear = nnear
        self.p = p
        self.exclude_nan = exclude_nan
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
            idw = Invdisttree(X=zip(x, y))
            self.idw = idw
            self.x = x
            self.y = y

        zi = idw(q=zip(xi, yi),
                 z=z,
                 nnear=self.nnear,
                 p=self.p)
        return zi


class OrdinaryKrigingInterpolator(PointsToGridInterpolator):
    def __init__(self,
                 nlags=30,
                 variogram_model='spherical',
                 weight=True,
                 n_closest_points=10,
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
                 interpolator=IdwKdtreeInterpolator()):

        self.lons, self.lats = get_lon_lat_list_from_cml_list(cml_list)
        # Later some coordinate transformations can be added here
        self.x = self.lons
        self.y = self.lats

        # TODO: Forward arguments to select aggregation type and frequency
        self.df_cmls = get_dataframe_for_cml_variable(cml_list)
        self._interpolator = interpolator
        self.resolution = resolution

        self.xgrid, self.ygrid = _generate_grid(x=self.x,
                                                y=self.y,
                                                xgrid=xgrid,
                                                ygrid=ygrid,
                                                resolution=resolution)

    def interpolate_for_i(self, i):
        z = self.df_cmls.iloc[i, :]
        i_not_nan = ~pd.isnull(z)

        if z[i_not_nan].sum() == 0:
            # print('%s: Returning NaNs because data contains only NaNs' %
            #      self.df_cmls.index[i])
            zgrid = np.zeros_like(self.xgrid)
            zgrid[:] = np.nan
        else:
            zgrid = self._interpolator(x=self.x,
                                       y=self.y,
                                       z=z,
                                       xgrid=self.xgrid,
                                       ygrid=self.ygrid)
        return zgrid

    def loop_over_time(self):
        zi_list = []

        for i in tqdm(range(len(self.df_cmls.index))):
            try:
                zi = self.interpolate_for_i(i)
            except ValueError as e:
                # Catch Kriging error and return NaNs
                if e.args[0] == 'Singular matrix':
                    print('%s: Kriging calculations produced '
                          'singular matrix. Returning NaNs.'
                          % self.df_cmls.index[i])
                    zi = np.ones_like(self._interpolator.xgrid.ravel())
                    zi[:] = np.nan
                    sigma = np.ones_like(self._interpolator.xgrid.ravel())
                    sigma[:] = np.nan
                else:
                    print 'baz'
                    raise e
            zi_list.append(zi)
        return zi_list


def _generate_grid(x, y, xgrid, ygrid, resolution):
    """ Generate grid with certain resolution """

    if (xgrid is None) or (ygrid is None):

        if resolution is None:
            raise ValueError('`resolution must be set if `xgrid` '
                             'or `ygrid` are None')

        xcoords = np.arange(min(x) - resolution,
                            max(x) + resolution,
                            resolution)
        ycoords = np.arange(min(y) - resolution,
                            max(y) + resolution,
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
                                   resample_label='right',
                                   variable='R',
                                   channels=['channel_1'],
                                   aggregation_func=np.mean,
                                   apply_factor=1):
    """ Build a DataFrame for a certain variable for all CMLs """

    # Resample time series of each CML
    df_cmls_R = pd.DataFrame()

    # TODO: Extend the code to be able to average over two channels if desired
    channel_name = channels[0]

    for cml in cml_list:
        df_cmls_R[cml.metadata['cml_id']] = (
            cml.channels[channel_name].data[variable]
            .resample(resample_to, label=resample_label)
            .apply(aggregation_func))
    df_cmls_R *= apply_factor

    return df_cmls_R


