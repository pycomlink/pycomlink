from __future__ import division
from builtins import zip
from builtins import object
from collections import namedtuple
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon

from pycomlink.util.maintenance import deprecated


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
        self.intersect_weights = calc_intersect_weights(
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
        grid_corners = _calc_grid_corners_for_center_location(grid)

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
        intersect, pixel_poly_list = calc_intersect_weights(
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


def calc_intersect_weights(
    x1_line,
    y1_line,
    x2_line,
    y2_line,
    x_grid,
    y_grid,
    grid_point_location="center",
    offset=None,
    return_pixel_poly_list=False,
):
    """Calculate intersecting weights for a line and a grid

    Calculate the intersecting weights for the line defined by `x1_line`,
    `y1_line`, `x2_line` and `y2_line` and the grid defined by the x- and y-
    grid points from `x_grid` and `y_grid`.

    Parameters
    ----------
    x1_line : float
    y1_line : float
    x2_line : float
    y2_line : float
    x_grid : 2D array
        x-coordinates of grid points
    y_grid : 2D array
        y-coordinates of grid points
    grid_point_location : str, optional
        The only option currently is `center` which assumes that the
        coordinates in `xr_ds` represent the centers of grid cells
    offset : float, optional
        The offset in units of the coordinates to constrain the calculation
        of intersection to a bounding box around the CML coordinates. The
        offset specifies by how much this bounding box will be larger then
        the width- and height-extent of the CML coordinates.
    return_pixel_poly_list : bool, optional
        If `True`, also return the list of shapely.Polygon objects which were
        used to calculate the intersection weights. Defaults to `False`.

    Returns
    -------

    intersect : array
        2D array of intersection weights with shape of the longitudes- and
        latitudes grid of `xr_ds`
    pixel_poly_list : list, optional
        List of shapely.Polygons which were used to calculate intersections

    """

    x_grid = x_grid.astype("float64")
    y_grid = y_grid.astype("float64")

    # grid = np.stack([xr_ds.longitudes.values, xr_ds.latitudes.values], axis=2)
    grid = np.stack([x_grid, y_grid], axis=2)

    # Get link coordinates for easy access
    # cml_coords = cml.get_coordinates()

    # Convert CML to shapely line
    link = LineString([(x1_line, y1_line), (x2_line, y2_line)])

    # Derive grid cell width to set bounding box offset
    ll_cell = grid[0, 1, 0] - grid[0, 0, 0]
    ul_cell = grid[-1, 1, 0] - grid[-1, 0, 0]
    lr_cell = grid[0, -1, 0] - grid[0, -2, 0]
    ur_cell = grid[-1, -1, 0] - grid[-1, -2, 0]
    offset_calc = max(ll_cell, ul_cell, lr_cell, ur_cell)

    # Set bounding box offset
    if offset is None:
        offset = offset_calc

    # Set bounding box
    x_max = max([x1_line, x2_line])
    x_min = min([x1_line, x2_line])
    y_max = max([y1_line, y2_line])
    y_min = min([y1_line, y2_line])
    # lon_grid = grid[:, :, 0]
    # lat_grid = grid[:, :, 1]
    bounding_box = ((x_grid > x_min - offset) & (x_grid < x_max + offset)) & (
        (y_grid > y_min - offset) & (y_grid < y_max + offset)
    )

    # Calculate polygon corners assuming that `grid` defines the center
    # of each grid cell
    if grid_point_location == "center":
        grid_corners = _calc_grid_corners_for_center_location(grid)
    elif grid_point_location == "lower_left":
        grid_corners = _calc_grid_corners_for_lower_left_location(grid)
    else:
        raise ValueError(
            "`grid_point_location` = %s not implemented" % grid_point_location
        )

    # Find intersection
    intersect = np.zeros([grid.shape[0], grid.shape[1]])
    pixel_poly_list = []
    # Iterate only over the indices within the bounding box and
    # calculate the intersect weigh for each pixel
    ix_in_bbox = np.where(bounding_box == True)
    for i, j in zip(ix_in_bbox[0], ix_in_bbox[1]):
        pixel_poly = Polygon(
            [
                grid_corners.ll_grid[i, j],
                grid_corners.lr_grid[i, j],
                grid_corners.ur_grid[i, j],
                grid_corners.ul_grid[i, j],
            ]
        )
        pixel_poly_list.append(pixel_poly)

        c = link.intersection(pixel_poly)
        if not c.is_empty:
            intersect[i][j] = c.length / link.length

    if return_pixel_poly_list:
        return intersect, pixel_poly_list
    else:
        return intersect


def _calc_grid_corners_for_center_location(grid):
    """

    Parameters
    ----------
    grid : array
        3D matrix holding x and y grids. Shape of `grid` must be
        (height, width, 2).

    Returns
    -------

    namedtuple with the grids for the four corners of the grid defined
    by points at the lower left corner

    """

    grid = grid.astype("float64")

    # Upper right
    ur_grid = np.zeros_like(grid)
    ur_grid[0:-1, 0:-1, :] = (grid[0:-1, 0:-1, :] + grid[1:, 1:, :]) / 2.0
    ur_grid[-1, :, :] = ur_grid[-2, :, :] + (ur_grid[-2, :, :] - ur_grid[-3, :, :])
    ur_grid[:, -1, :] = ur_grid[:, -2, :] + (ur_grid[:, -2, :] - ur_grid[:, -3, :])
    # Upper left
    ul_grid = np.zeros_like(grid)
    ul_grid[0:-1, 1:, :] = (grid[0:-1, 1:, :] + grid[1:, :-1, :]) / 2.0
    ul_grid[-1, :, :] = ul_grid[-2, :, :] + (ul_grid[-2, :, :] - ul_grid[-3, :, :])
    ul_grid[:, 0, :] = ul_grid[:, 1, :] - (ul_grid[:, 2, :] - ul_grid[:, 1, :])
    # Lower right
    lr_grid = np.zeros_like(grid)
    lr_grid[1:, 0:-1, :] = (grid[1:, 0:-1, :] + grid[:-1, 1:, :]) / 2.0
    lr_grid[0, :, :] = lr_grid[1, :, :] - (lr_grid[2, :, :] - lr_grid[1, :, :])
    lr_grid[:, -1, :] = lr_grid[:, -2, :] + (lr_grid[:, -2, :] - lr_grid[:, -3, :])
    # Lower left
    ll_grid = np.zeros_like(grid)
    ll_grid[1:, 1:, :] = (grid[1:, 1:, :] + grid[:-1, :-1, :]) / 2.0
    ll_grid[0, :, :] = ll_grid[1, :, :] - (ll_grid[2, :, :] - ll_grid[1, :, :])
    ll_grid[:, 0, :] = ll_grid[:, 1, :] - (ll_grid[:, 2, :] - ll_grid[:, 1, :])

    GridCorners = namedtuple(
        "GridCorners", ["ur_grid", "ul_grid", "lr_grid", "ll_grid"]
    )

    return GridCorners(
        ur_grid=ur_grid, ul_grid=ul_grid, lr_grid=lr_grid, ll_grid=ll_grid
    )


def _calc_grid_corners_for_lower_left_location(grid):
    """

    Parameters
    ----------
    grid : array
        3D matrix holding x and y grids. Shape of `grid` must be
        (height, width, 2).

    Returns
    -------

    namedtuple with the grids for the four corners around the
    central grid points

    """

    grid = grid.astype("float64")

    if (np.diff(grid[:, :, 0], axis=1) < 0).any():
        raise ValueError("x values must be ascending along axis 1")
    if (np.diff(grid[:, :, 1], axis=0) < 0).any():
        raise ValueError("y values must be ascending along axis 0")

    # Upper right
    ur_grid = np.zeros_like(grid)
    ur_grid[0:-1, 0:-1, :] = grid[1:, 1:, :]
    ur_grid[-1, :, :] = ur_grid[-2, :, :] + (ur_grid[-2, :, :] - ur_grid[-3, :, :])
    ur_grid[:, -1, :] = ur_grid[:, -2, :] + (ur_grid[:, -2, :] - ur_grid[:, -3, :])
    # Upper left
    ul_grid = np.zeros_like(grid)
    ul_grid[0:-1, 0:-1, :] = grid[1:, 0:-1, :]
    ul_grid[-1, :, :] = ul_grid[-2, :, :] + (ul_grid[-2, :, :] - ul_grid[-3, :, :])
    ul_grid[:, -1, :] = ul_grid[:, -2, :] + (ul_grid[:, -2, :] - ul_grid[:, -3, :])
    # Lower right
    lr_grid = np.zeros_like(grid)
    lr_grid[0:-1, 0:-1, :] = grid[0:-1, 1:, :]
    lr_grid[-1, :, :] = lr_grid[-2, :, :] + (lr_grid[-2, :, :] - lr_grid[-3, :, :])
    lr_grid[:, -1, :] = lr_grid[:, -2, :] + (lr_grid[:, -2, :] - lr_grid[:, -3, :])
    # Lower left
    ll_grid = grid.copy()

    GridCorners = namedtuple(
        "GridCorners", ["ur_grid", "ul_grid", "lr_grid", "ll_grid"]
    )

    return GridCorners(
        ur_grid=ur_grid, ul_grid=ul_grid, lr_grid=lr_grid, ll_grid=ll_grid
    )


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
