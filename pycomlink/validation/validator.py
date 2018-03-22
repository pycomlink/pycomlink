from __future__ import division
from builtins import zip
from builtins import object
import xarray as xr
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Polygon

from pycomlink.util.maintenance import deprecated

class Validator(object):
    def __init__(self):
        pass

    def calc_stats(self, cml, time_series):
        cml_copy = cml
        time_series_df = pd.DataFrame(data={'xr_ds': time_series},
                                      index=time_series.time)
        cml_copy.data.index = cml_copy.data.index.strftime("%Y-%m-%d %H:%M:%S")
        # pd.DatetimeIndex(cml.data.index)
        joined_df = time_series_df.join(cml_copy.data.txrx_nf)
        pearson_r = joined_df.xr_ds.corr(joined_df.txrx_nf, method='pearson')
        return pearson_r

    def _get_cml_pair(self, cml):
        pass


class GridValidator(Validator):
    def __init__(self, lats=None, lons=None, values=None, xr_ds=None):
        if xr_ds is None:
            # construct xr_ds from lats, lons & values here?
            xr_ds = xr.Dataset()

        self.xr_ds = xr_ds
        self.intersect_weights = None
        self.weighted_grid_sum = None

        pass

    def _get_cml_pair_indices(self, cml):
        self.cml = cml
        # get intersect weights
        self.intersect_weights = calc_intersect_weights(self.cml, self.xr_ds)
        # get weighted truth values from grid

        return self.intersect_weights

    def get_time_series(self, cml, values):
        intersect_weights = self._get_cml_pair_indices(cml)

        # Get start and end time of CML data set to constrain lookup in `xr_ds`
        t_start = cml.channel_1.data.index.values[0]
        t_stop = cml.channel_2.data.index.values[-1]

        t_mask = (self.xr_ds.time > t_start) & (self.xr_ds.time < t_stop)

        # Get bounding box where CML intersects with grid to constrain
        # lookup in `xr_ds`
        w_mask = intersect_weights > 0
        # Since we cannot use a 2D mask in xarray, build the slices of
        # indices for the x- and y-axis
        w_ix_x = np.unique(np.where(w_mask)[0])
        w_ix_y = np.unique(np.where(w_mask)[1])
        slice_x = slice(w_ix_x.min(), w_ix_x.max()+1)
        slice_y = slice(w_ix_y.min(), w_ix_y.max()+1)

        self.weighted_grid_sum = (self.xr_ds[values][t_mask, slice_x, slice_y] *
                                  intersect_weights[slice_x, slice_y]
                                  ).sum(dim=['x', 'y']).to_dataframe()

        return self.weighted_grid_sum

    def resample_to_grid_time_series(self,
                                     df,
                                     grid_time_index_label,
                                     grid_time_zone=None):
        df_temp = df.copy()
        df_truth_t = pd.DataFrame(self.weighted_grid_sum.index,
                                  self.weighted_grid_sum.index)
        if grid_time_zone is not None:
            df_truth_t.index = df_truth_t.index.tz_localize(grid_time_zone)
        if grid_time_index_label == 'right':
            method = 'bfill'
        elif grid_time_index_label == 'left':
            method = 'ffill'
        else:
            raise NotImplementedError('Only `left` and `right` are allowed up '
                                      'to now for `grid_time_index_label.')
        df_truth_t = df_truth_t.reindex(df.index, method=method)
        df_temp['truth_time_ix'] = df_truth_t.time

        return df_temp.groupby('truth_time_ix').mean()


class PointValidator(Validator):
    def __init__(lats, lons, values):
        self.truth_data = [lats, lons, time_series]

    def _get_cml_pair_indices(cml):
        # get nearest point location

        return pair_indices

    def get_time_series(self, cml, values):
        pass


def calc_intersect_weights(cml,
                           xr_ds,
                           grid_point_location='center',
                           offset=None):
    """

    Parameters
    ----------
    cml : Comlink
        One Comlink object for whose path the intersecting weights should
        be calculated
    xr_ds : xarray.DataSet or xarrat.DataArray
        DataSete or DataArray that must have coordinates named `longitudes`
        and `latitudes` which are each a 2D matrices of the coordinates
    grid_point_location : str, optional
        The only option currently is `center` which assumes that the
        coordinates in `xr_ds` represent the centers of grid cells
    offset : float, optional
        The offset in units of the coordinates to contrain the calculation
        of intersection to a bounding box around the CML coordinates. The
        offset specifies by how much this bounding box will be larger then
        the widht- and height-extent of the CML coordinates.

    Returns
    -------

    intersect : array
        2D array of intersection weights with shape of the longitudes- and
        latitudes grid of `xr_ds`

    """
    grid = np.stack([xr_ds.longitudes.values, xr_ds.latitudes.values], axis=2)

    # Get link coordinates for easy access
    cml_coords = cml.get_coordinates()

    # Convert CML to shapely line
    link = LineString([(cml_coords.lon_a, cml_coords.lat_a),
                       (cml_coords.lon_b, cml_coords.lat_b)])

    # Derive grid cell width to set bounding box offset
    ll_cell = grid[0, 1, 0] - grid[0, 0, 0]
    ul_cell = grid[-1, 1, 0] - grid[-1, 0, 0]
    lr_cell = grid[0, -1, 0] - grid[0, -2, 0]
    ur_cell = (grid[-1, -1, 0] - grid[-1, -2, 0])
    offset_calc = max(ll_cell, ul_cell, lr_cell, ur_cell)

    # Set bounding box offset
    if offset is None:
        offset = offset_calc

    # Set bounding box
    lon_max = max([cml_coords.lon_a, cml_coords.lon_b])
    lon_min = min([cml_coords.lon_a, cml_coords.lon_b])
    lat_max = max([cml_coords.lat_a, cml_coords.lat_b])
    lat_min = min([cml_coords.lat_a, cml_coords.lat_b])
    lon_grid = grid[:, :, 0]
    lat_grid = grid[:, :, 1]
    bounding_box = (
        ((lon_grid > lon_min - offset) & (lon_grid < lon_max + offset)) &
        ((lat_grid > lat_min - offset) & (lat_grid < lat_max + offset)))

    # Calculate polygon corners assuming that `grid` defines the center
    # of each grid cell
    #
    # Upper right
    ur_grid = np.zeros_like(grid)
    ur_grid[0:-1, 0:-1, :] = (grid[0:-1, 0:-1, :] + grid[1:, 1:, :]) / 2.0
    ur_grid[-1, :, :] = (ur_grid[-2, :, :]
                         + (ur_grid[-2, :, :] - ur_grid[-3, :, :]))
    ur_grid[:, -1, :] = (ur_grid[:, -2, :]
                         + (ur_grid[:, -2, :] - ur_grid[:, -3, :]))
    # Upper left
    ul_grid = np.zeros_like(grid)
    ul_grid[0:-1, 1:, :] = (grid[0:-1, 1:, :] + grid[1:, :-1, :]) / 2.0
    ul_grid[-1, :, :] = (ul_grid[-2, :, :]
                         + (ul_grid[-2, :, :] - ul_grid[-3, :, :]))
    ul_grid[:, 0, :] = (ul_grid[:, 1, :]
                        - (ul_grid[:, 2, :] - ul_grid[:, 1, :]))
    # Lower right
    lr_grid = np.zeros_like(grid)
    lr_grid[1:, 0:-1, :] = (grid[1:, 0:-1, :] + grid[:-1, 1:, :]) / 2.0
    lr_grid[0, :, :] = (lr_grid[1, :, :]
                        - (lr_grid[2, :, :] - lr_grid[1, :, :]))
    lr_grid[:, -1, :] = (lr_grid[:, -2, :]
                         + (lr_grid[:, -2, :] - lr_grid[:, -3, :]))
    # Lower left
    ll_grid = np.zeros_like(grid)
    ll_grid[1:, 1:, :] = (grid[1:, 1:, :] + grid[:-1, :-1, :]) / 2.0
    ll_grid[0, :, :] = (ll_grid[1, :, :]
                        - (ll_grid[2, :, :] - ll_grid[1, :, :]))
    ll_grid[:, 0, :] = (ll_grid[:, 1, :]
                        - (ll_grid[:, 2, :] - ll_grid[:, 1, :]))

    # Find intersection
    intersect = np.zeros([grid.shape[0], grid.shape[1]])
    pixel_poly_list = []
    # Iterate only over the indices within the bounding box and
    # calculate the intersect weigh for each pixel
    ix_in_bbox = np.where(bounding_box == True)
    for i, j in zip(ix_in_bbox[0], ix_in_bbox[1]):
        if grid_point_location == 'center':
            pixel_poly = Polygon(
                [ll_grid[i, j],
                 lr_grid[i, j],
                 ur_grid[i, j],
                 ul_grid[i, j]])
            pixel_poly_list.append(pixel_poly)
        else:
            raise ValueError('`grid_point_location` = %s not implemented' %
                             grid_point_location)

        c = link.intersection(pixel_poly)
        if not c.is_empty:
            intersect[i][j] = (c.length / link.length)
    return intersect


@deprecated('Use `pycomlink.validation.stats.calc_wet_error_rates()` '
            'instead since the `dry_error` here makes no sense.')
def calc_wet_dry_error(df_wet_truth, df_wet):
    dry_error = (((df_wet_truth == False) & (df_wet == True)).sum() /
                 float((df_wet_truth == False).sum()))
    wet_error = (((df_wet_truth == True) & (df_wet == False)).sum() /
                 float((df_wet_truth == True).sum()))
    return wet_error, dry_error
