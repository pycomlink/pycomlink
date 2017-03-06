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


import scipy
import numpy as np
import pandas as pd
import xarray as xr

from ok import kriging


class Interpolator(object):
    def __init__(self,
                 cml_list,
                 grid=None,
                 resolution=None,
                 resample_time='H',
                 resample_func='mean',
                 resample_label='right',
                 variable='R'):
        self.cml_list = cml_list
        self.variable = variable

        self.gridded_data = None

        if grid is None:
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

            self.grid = np.vstack((xi, yi)).T
            self.xgrid = xgrid
            self.ygrid = ygrid
        else:
            self.grid = grid

        # Resample time series of each CML
        self.df_cmls_R = pd.DataFrame()
        # TODO: Add options for selecting channels and/or average them
        for cml in self.cml_list:
            self.df_cmls_R[cml.metadata['cml_id']] = (
                cml.channel_1.data[self.variable]
                .resample(resample_time, label=resample_label)
                .apply(resample_func))

        # Extract lats and lons
        self.lons = np.array(
            [cml.get_center_lon_lat()[0] for cml in self.cml_list])
        self.lats = np.array(
            [cml.get_center_lon_lat()[1] for cml in self.cml_list])

    def kriging(self, n_closest_points):
        # TODO: FIX Kriging. Results do not yet make sense...
        fields = []

        for t, row in self.df_cmls_R.iterrows():
            values = row.values
            i_not_nan = ~pd.isnull(values)
            interp_values = kriging(self.lons[i_not_nan],
                                    self.lats[i_not_nan],
                                    values,
                                    self.xgrid,
                                    self.ygrid,
                                    n_closest_points=n_closest_points)
            fields.append(interp_values)

        self.gridded_data = self._fields_to_dataset(fields)
        return self.gridded_data

    def idw(self, max_dist=None, power=2):
        fields = []

        for t, row in self.df_cmls_R.iterrows():
            values = row.values
            i_not_nan = ~pd.isnull(values)
            interp_values = idw(values[i_not_nan],
                                self.lons[i_not_nan],
                                self.lats[i_not_nan],
                                self.xgrid,
                                self.ygrid,
                                max_dist=None,
                                p=power)
            fields.append(interp_values)

        self.gridded_data = self._fields_to_dataset(fields)
        return self.gridded_data

    def idw_kdtree(self):
        fields = []

        for t, row in self.df_cmls_R.iterrows():
            values = row.values
            i_not_nan = ~pd.isnull(values)
            idw_tree = Invdisttree(zip(self.lats[i_not_nan],
                                       self.lons[i_not_nan]),
                                   values[i_not_nan])
            interp_values = idw_tree(zip(self.xgrid.flatten(),
                                         self.ygrid.flatten()),
                                     k=6,
                                     eps=0)
            interp_values = np.reshape(interp_values, self.xgrid.shape)
            fields.append(interp_values)

        self.gridded_data = self._fields_to_dataset(fields)
        return self.gridded_data

    def rbf(self):
        from scipy.interpolate import Rbf

        fields = []
        for t, row in self.df_cmls_R.iterrows():
            values = row.values
            i_not_nan = ~pd.isnull(values)

            rbf = Rbf(self.lons[i_not_nan],
                      self.lats[i_not_nan],
                      values[i_not_nan],
                      function='linear')
            interp_values = rbf(self.xgrid, self.ygrid)
            fields.append(interp_values)

        self.gridded_data = self._fields_to_dataset(fields)
        return self.gridded_data

    def _fields_to_dataset(self, fields):
        ds = xr.Dataset({self.variable: (['x', 'y', 'time'],
                                         np.moveaxis(np.array(fields),
                                                             0, -1))},
                        coords={'lon': (['x', 'y'], self.xgrid),
                                'lat': (['x', 'y'], self.ygrid),
                                'time': self.df_cmls_R.index})
        return ds


def idw(z, x, y, xi, yi, max_dist=None, p=2):

    # Code adapted from
    # http://stackoverflow.com/questions/3104781/...
    # inverse-distance-weighted-idw-interpolation-with-python

    ny, nx = xi.shape

    dist = distance_matrix(x,y, xi.flatten(), yi.flatten())

    if max_dist is not None:
        dist[dist > max_dist] = 0

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist**p

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)

    zi = zi.reshape((ny,nx))

    return zi


import numpy as np
from scipy.spatial import cKDTree as KDTree




class Invdisttree:
        """ inverse-distance-weighted interpolation using KDTree:
        invdisttree = Invdisttree( X, z )  -- points, values
        interpol = invdisttree( q, k=6, eps=0 )
            -- interpolate z from the 6 points nearest each q;
               q may be one point, or a batch of points
        """
        def __init__(self, X, z, leafsize=10 ):
            self.tree = KDTree(X, leafsize=leafsize)  # build the tree
            self.z = z

        def __call__(self, q, k=6, eps=0 ):
            # k nearest neighbours of each query point --
            self.distances, self.ix = self.tree.query( q, k=k, eps=eps )
            interpol = []  # np.zeros( (len(self.distances),) + np.shape(z[0]) )
            for dist, ix in zip( self.distances, self.ix ):
                if dist[0] > 1e-10:
                    w = 1 / dist
                    wz = np.dot( w, self.z[ix] ) / np.sum(w)  # weightz s by 1/dist
                else:
                    wz = self.z[ix[0]]
                interpol.append( wz )
            return interpol


def distance_matrix(x0, y0, x1, y1):
    # Code adapted from
    # http://stackoverflow.com/questions/3104781/...
    # inverse-distance-weighted-idw-interpolation-with-python

    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])

    return np.hypot(d0, d1)
