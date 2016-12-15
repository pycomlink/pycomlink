import xarray as xr
import pandas as pd
import numpy as np

from shapely.geometry import LineString, Polygon


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

        pass

    def _get_cml_pair_indices(self, cml):
        self.cml = cml
        # get intersect weights
        self.intersect_weights = calc_intersect_weights(self.cml, self.xr_ds)
        # get weighted truth values from grid

        return self.intersect_weights

    def get_time_series(self, cml, values):
        intersect = self._get_cml_pair_indices(cml)

        grid_sum = np.zeros([len(self.xr_ds.time)])
        for i in range(len(intersect)):
            for j in range(len(intersect)):
                if intersect[i][j] != 0:
                    grid_sum = grid_sum + (
                        intersect[i][j] * self.xr_ds[values][:, i, j])

        return grid_sum


class PointValidator(Validator):
    def __init__(lats, lons, values):
        self.truth_data = [lats, lons, time_series]

    def _get_cml_pair_indices(cml):
        # get nearest point location

        return pair_indices

    def get_time_series(self, cml, values):
        pass


def calc_intersect_weights(cml, xr_ds, offset=None):
    grid = np.stack([xr_ds.longitudes.values, xr_ds.latitudes.values], axis=2)

    # Get link coordinates for easy access
    lonA = cml.metadata['site_A']['lon']
    lonB = cml.metadata['site_B']['lon']
    latA = cml.metadata['site_A']['lat']
    latB = cml.metadata['site_B']['lat']

    # Convert CML to shapely line
    link = LineString([(lonA, latA), (lonB, latB)])

    # Derive grid cell width to set bounding box offset
    ll_cell = grid[0, 1, 0] - grid[0, 0, 0]
    ul_cell = grid[(len(grid) - 1), 1, 0] - grid[(len(grid) - 1), 0, 0]
    lr_cell = grid[0, (len(grid) - 1), 0] - grid[0, (len(grid) - 2), 0]
    ur_cell = grid[(len(grid) - 1), (len(grid) - 1), 0] - grid[
        (len(grid) - 1), (len(grid) - 2), 0]
    offset_calc = max(ll_cell, ul_cell, lr_cell, ur_cell)

    # Set bounding box offset
    if offset is None:
        offset = offset_calc

    # Set bounding box
    lon_max = max([lonA, lonB])
    lon_min = min([lonA, lonB])
    lat_max = max([latA, latB])
    lat_min = min([latA, latB])
    lon_grid = grid[:, :, 0]
    lat_grid = grid[:, :, 1]
    bounding_box = (
        ((lon_grid > lon_min - offset) & (lon_grid < lon_max + offset)) &
        ((lat_grid > lat_min - offset) & (lat_grid < lat_max + offset)))

    # Find intersection
    intersect = np.zeros([grid.shape[0], grid.shape[1]])
    for i in range(0, grid.shape[0] - 1):
        for j in range(0, grid.shape[1] - 1):
            if bounding_box[i, j] == True:
                poly = [(grid[i, j]), (grid[i + 1, j]),
                        (grid[i + 1, j + 1]), (grid[i, j + 1])]
                pixel = Polygon(poly)
                c = link.intersection(pixel)
                if not c.is_empty:
                    intersect[i][j] = (c.length / link.length)
    return intersect
