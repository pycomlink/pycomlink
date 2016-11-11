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


class Interpolator(object):
    def __init__(self, cml_list, grid=None, resolution=None):
        self.cml_list = cml_list

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

    def idw(self, resample_time, resample_func, power=2):

        self.fields = []

        df_cmls_R = pd.DataFrame()
        for cml in self.cml_list:
            df_cmls_R[cml.metadata['cml_id']] = \
                cml.channel_1._df['R'].resample(
                    resample_time).apply(resample_func)


        lons = []
        lats = []
        sample_values = []
        for time, df_row in df_cmls_R.iterrows():
            for cml in self.cml_list:
                lons.append(cml.get_center_lon_lat()[0])
                lats.append(cml.get_center_lon_lat()[1])
                sample_values.append(df_row[cml.metadata['cml_id']])

            interp_values = idw(sample_values,
                                lons,
                                lats,
                                self.xgrid,
                                self.ygrid,
                                p=power)
            self.fields.append(interp_values)


def idw(z, x, y, xi, yi, p=2):

    # Code adapted from
    # http://stackoverflow.com/questions/3104781/...
    # inverse-distance-weighted-idw-interpolation-with-python

    ny, nx = xi.shape

    dist = distance_matrix(x,y, xi.flatten(), yi.flatten())

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist**p

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)

    zi = zi.reshape((ny,nx))

    return zi


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
