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

from __future__ import division
from __future__ import print_function

from builtins import object
import numpy as np
import pandas as pd
import xarray as xr

from pykrige.ok import OrdinaryKriging

from tqdm import tqdm

from .idw import Invdisttree
from ..util.maintenance import deprecated


@deprecated('Use the new `Interpolator` classes from '
            '`pycomlink.spatial.interpolator`.')
class Interpolator(object):
    """ Class for interpolating CML data onto a grid using different methods

    The CML data, typically the processed rain rate, is resampled to a
    defined time interval and stored as pandas.DataFrame. The gridded data
    after interpolation is stored an a xarray.Dataset.

    Parameters
    ----------

    cml_list : list
        List of Comlink objects
    channel_name : str, optional
        Key of the channel to use. Defaults to 'channel_1'
    xgrid : array_like, optional
        2D grid of x-coordinates
    ygrid : array_like, optional
        2D grid of y-coordinates
    resolution : float, optional
        Resolution of grid that will be generated and used for interpolation,
        based on the bounding box around all CMLs in `cml_list`.
    resample_time : str, optional
        Resampling time for CML data. xarray nomenclature is used.
        Defaults to 'H'.
    resample_func : str, optional
        Function to use for resampling, as understood by  xarray.
        Defaults to 'mean'.
    resample_label : str, optional
        Position of temporal label to use while resampling. Defaults 'right'.
    apply_factor : numeric, optional
        Factor that is applied to the resampled data. Defaults to 1.
    variable : str, optional
        Variable name in the pandas.DataFrame of the ComlinkChannel.data that
        will be used for resampling an interpolation.
    """

    def __init__(self,
                 cml_list,
                 channel_name='channel_1',
                 xgrid=None,
                 ygrid=None,
                 resolution=None,
                 resample_time='H',
                 resample_func='mean',
                 resample_label='right',
                 apply_factor=1,
                 variable='R'):

        self.cml_list = cml_list
        self.variable = variable

        self.gridded_data = None
        self.grid_points_covered_by_cmls = None

        if (xgrid is None) or (ygrid is None):
            lats = ([cml.metadata['site_a_latitude'] for cml in cml_list] +
                    [cml.metadata['site_b_latitude'] for cml in cml_list])
            lons = ([cml.metadata['site_a_longitude'] for cml in cml_list] +
                    [cml.metadata['site_b_longitude'] for cml in cml_list])

            if resolution is None:
                resolution = 0.01

            xcoords = np.arange(min(lons) - resolution,
                                max(lons) + resolution,
                                resolution)
            ycoords = np.arange(min(lats) - resolution,
                                max(lats) + resolution,
                                resolution)
            xgrid, ygrid = np.meshgrid(xcoords, ycoords)
            xi, yi = xgrid.flatten(), ygrid.flatten()

            self.xgrid = xgrid
            self.ygrid = ygrid
        else:
            self.xgrid = xgrid
            self.ygrid = ygrid

        # Resample time series of each CML
        self.df_cmls_R = pd.DataFrame()
        # TODO: Add options average results for several channels
        for cml in self.cml_list:
            self.df_cmls_R[cml.metadata['cml_id']] = (
                cml.channels[channel_name].data[self.variable]
                .resample(resample_time, label=resample_label)
                .apply(resample_func))
        self.df_cmls_R *= apply_factor

        # Extract lats and lons
        self.lons = np.array(
            [cml.get_center_lon_lat()[0] for cml in self.cml_list])
        self.lats = np.array(
            [cml.get_center_lon_lat()[1] for cml in self.cml_list])

    def kriging(self,
                progress_bar=False,
                t_start=None, t_stop=None):
        fields = []

        if t_start is None:
            t_start = self.df_cmls_R.index[0]
        if t_stop is None:
            t_stop = self.df_cmls_R.index[-1]

        if progress_bar:
            pbar = tqdm(total=len(self.df_cmls_R[t_start:t_stop].index))

        for t, row in self.df_cmls_R[t_start:t_stop].iterrows():
            values = row.values
            i_not_nan = ~pd.isnull(values)

            if values[i_not_nan].sum() == 0:
                print('Skipping %s' % t)
                zi = np.zeros_like(self.xgrid)

            else:
                try:
                    ok = OrdinaryKriging(x=self.lons[i_not_nan],
                                         y=self.lats[i_not_nan],
                                         z=values[i_not_nan],
                                         nlags=30,
                                         variogram_model='spherical',
                                         weight=True)

                    zi, sigma = ok.execute('points',
                                           self.xgrid.flatten(),
                                           self.ygrid.flatten(),
                                           n_closest_points=10,
                                           backend='C')
                    zi = np.reshape(zi, self.xgrid.shape)
                except:
                    #if 'Singular matrix' in err.message:
                    #    print 'Singular matrix encountered while doing ' \
                    #          'moving window kriging.'
                    print('Error while doing kriging for %s' % t)
                    zi = np.zeros_like(self.xgrid)
                    #else:
                    #    raise

            fields.append(zi)

            if progress_bar:
                pbar.update(1)

        # Close progress bar
        if progress_bar:
            pbar.close()

        self.gridded_data = self._fields_to_dataset(fields)
        return self.gridded_data

    def idw_kdtree(self, nnear=10, p=2, eps=0.1,
                   progress_bar=False,
                   t_start=None, t_stop=None):
        """ Perform Inverse Distance Weighting interpolation using a kd-tree

        Parameters
        ----------

        nnear : int, optional
            Number of nearest neighbours. Defaults to 10.
        p : int, optional
            Exponent in 1/r**p used to derived the weights for interpolation
        eps : float, optional
            Approximate nearest neighbours so that
            distance <= (1 + eps) * true nearest
        progress_bar : bool, optional
            Switch on/off progress for loop of time steps
        t_start : numpy.datetime64, datetime, or str using pandas syntax
            Time at which to start with the interpolation
        t_stop : numpy.datetime64, datetime, or str using pandas syntax
            Time at which to stop with the interpolation

        Returns
        -------

        xarray.Dataset with gridded data

        """

        fields = []

        if t_start is None:
            t_start = self.df_cmls_R.index[0]
        if t_stop is None:
            t_stop = self.df_cmls_R.index[-1]

        if progress_bar:
            pbar = tqdm(total=len(self.df_cmls_R[t_start:t_stop].index))

        for t, row in self.df_cmls_R[t_start:t_stop].iterrows():
            values = row.values
            i_not_nan = ~pd.isnull(values)

            idw_tree = Invdisttree(X=np.array([self.lons[i_not_nan],
                                               self.lats[i_not_nan]]).T,
                                   leafsize=nnear+2)
            interp_values = idw_tree(q=np.array([self.xgrid.flatten(),
                                                 self.ygrid.flatten()]).T,
                                     z=values[i_not_nan],
                                     nnear=nnear,
                                     p=p,
                                     eps=eps)
            interp_values = np.reshape(interp_values, self.xgrid.shape)
            fields.append(interp_values)

            if progress_bar:
                pbar.update(1)

        # Close progress bar
        if progress_bar:
            pbar.close()

        self.gridded_data = self._fields_to_dataset(fields, t_start, t_stop)
        return self.gridded_data

    def _fields_to_dataset(self, fields, t_start=None, t_stop=None):
        if t_start is None:
            t_start = self.df_cmls_R.index[0]
        if t_stop is None:
            t_stop = self.df_cmls_R.index[-1]

        ds = xr.Dataset({self.variable: (['x', 'y', 'time'],
                                         np.moveaxis(np.array(fields),
                                                             0, -1))},
                        coords={'lon': (['x', 'y'], self.xgrid),
                                'lat': (['x', 'y'], self.ygrid),
                                'time': (self.df_cmls_R[t_start:t_stop]
                                         .index.values
                                         .astype(np.datetime64))})
        return ds

