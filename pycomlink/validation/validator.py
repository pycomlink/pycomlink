from __future__ import division
from builtins import zip
from builtins import object
from collections import namedtuple
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
import poligrain as plg

from .. util.maintenance import deprecated

class Validator(object):
    def __init__(self):
        pass

    def calc_stats(self, cml, time_series):
        cml_copy = cml
        time_series_df = pd.DataFrame(
            data={"xr_ds": time_series}, index=time_series.time
        )
        cml_copy.data.index = cml_copy.data.index.strftime("%Y-%m-%d %H:%M:%S")
        # pd.DatetimeIndex(cml.data.index)
        joined_df = time_series_df.join(cml_copy.data.txrx_nf)
        pearson_r = joined_df.xr_ds.corr(joined_df.txrx_nf, method="pearson")
        return pearson_r


class GridValidator(Validator):
    def __init__(self, lats=None, lons=None, values=None, xr_ds=None):
        if xr_ds is None:
            # construct xr_ds from lats, lons & values here?
            xr_ds = xr.Dataset()

        self.xr_ds = xr_ds
        self.intersect_weights = None
        self.weighted_grid_sum = None

        pass

    def _get_cml_intersection_weights(self, cml):
        self.cml = cml
        cml_coords = cml.get_coordinates()

        # get intersect weights
        self.intersect_weights = plg.spatial.calc_intersect_weights(
            x1_line=cml_coords.lon_a,
            y1_line=cml_coords.lat_a,
            x2_line=cml_coords.lon_b,
            y2_line=cml_coords.lat_b,
            x_grid=self.xr_ds.longitudes.values,
            y_grid=self.xr_ds.latitudes.values,
        )

        return self.intersect_weights

    def get_time_series(self, cml, values):
        intersect_weights = self._get_cml_intersection_weights(cml)

        # Get start and end time of CML data set to constrain lookup in `xr_ds`
        t_start = cml.channel_1.data.index.values[0]
        t_stop = cml.channel_2.data.index.values[-1]

        t_mask = (self.xr_ds.time > t_start) & (self.xr_ds.time < t_stop)

        # Get bounding box where CML intersects with grid to constrain
        # lookup in `xr_ds`
        w_mask = intersect_weights > 0
        # Since we cannot use a 2D mask in xarray, build the slices of
        # indices for the x- and y-axis
        w_ix_x = np.unique(np.where(w_mask)[1])
        w_ix_y = np.unique(np.where(w_mask)[0])
        slice_x = slice(w_ix_x.min(), w_ix_x.max() + 1)
        slice_y = slice(w_ix_y.min(), w_ix_y.max() + 1)

        self.weighted_grid_sum = (
            (
                self.xr_ds[values][t_mask, slice_y, slice_x]
                * intersect_weights[slice_y, slice_x]
            )
            .sum(dim=["x", "y"])
            .to_dataframe()
        )

        return self.weighted_grid_sum

    def resample_to_grid_time_series(
        self, df, grid_time_index_label, grid_time_zone=None
    ):
        df_temp = df.copy()
        df_truth_t = pd.DataFrame(
            self.weighted_grid_sum.index, self.weighted_grid_sum.index
        )
        if grid_time_zone is not None:
            df_truth_t.index = df_truth_t.index.tz_localize(grid_time_zone)
        if grid_time_index_label == "right":
            method = "bfill"
        elif grid_time_index_label == "left":
            method = "ffill"
        else:
            raise NotImplementedError(
                "Only `left` and `right` are allowed up "
                "to now for `grid_time_index_label."
            )
        df_truth_t = df_truth_t.reindex(df.index, method=method)
        df_temp["truth_time_ix"] = df_truth_t.time

        return df_temp.groupby("truth_time_ix").mean()

    def plot_intersections(self, cml, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        cml.plot_line(ax=ax)

        # Generate lon-lat grid assuming the original coordinates represent
        # the center of the grid
        lons = np.zeros(self.xr_ds.longitudes.shape + np.array([1, 1]))
        lats = np.zeros(self.xr_ds.latitudes.shape + np.array([1, 1]))

        grid = np.stack(
            [self.xr_ds.longitudes.values, self.xr_ds.latitudes.values], axis=2
        )
        grid_corners = plg.spatial._calc_grid_corners_for_center_location(grid)

        lons[:-1, :-1] = grid_corners.ll_grid[:, :, 0]
        lons[-1, :-1] = grid_corners.ul_grid[-1, :, 0]
        lons[:-1, -1] = grid_corners.lr_grid[:, -1, 0]
        lons[-1, -1] = grid_corners.ur_grid[-1, -1, 0]

        lats[:-1, :-1] = grid_corners.ll_grid[:, :, 1]
        lats[-1, :-1] = grid_corners.ul_grid[-1, :, 1]
        lats[:-1, -1] = grid_corners.lr_grid[:, -1, 1]
        lats[-1, -1] = grid_corners.ur_grid[-1, -1, 1]

        cml_coords = cml.get_coordinates()

        # get intersect weights
        intersect, pixel_poly_list = plg.spatial.calc_intersect_weights(
            x1_line=cml_coords.lon_a,
            y1_line=cml_coords.lat_a,
            x2_line=cml_coords.lon_b,
            y2_line=cml_coords.lat_b,
            x_grid=self.xr_ds.longitudes.values,
            y_grid=self.xr_ds.latitudes.values,
            return_pixel_poly_list=True,
        )

        ax.pcolormesh(lons, lats, intersect, cmap=plt.cm.gray_r)
        ax.scatter(
            self.xr_ds.longitudes.values, self.xr_ds.latitudes.values, s=1, c="k"
        )
        for pixel_poly in pixel_poly_list:
            ax.plot(*pixel_poly.exterior.xy)

        ax.set_ylim()

        return ax


class PointValidator(Validator):
    def __init__(lats, lons, values):
        # self.truth_data = [lats, lons, time_series]
        pass

    def _get_cml_pair_indices(cml):
        # get nearest point location

        # return pair_indices
        pass

    def get_time_series(self, cml, values):
        pass


@deprecated(
    "Use `pycomlink.validation.stats.calc_wet_error_rates()` "
    "instead since the `dry_error` here makes no sense."
)
def calc_wet_dry_error(df_wet_truth, df_wet):
    dry_error = ((df_wet_truth == False) & (df_wet == True)).sum() / float(
        (df_wet_truth == False).sum()
    )
    wet_error = ((df_wet_truth == True) & (df_wet == False)).sum() / float(
        (df_wet_truth == True).sum()
    )
    return wet_error, dry_error
