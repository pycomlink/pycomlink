import shapely as sh
import geopandas


def calc_coverage_mask(cml_list, xgrid, ygrid, max_dist_from_cml):
        """ Generate a coverage mask with a certain area around all CMLs

        Parameters
        ----------

        cml_list : list
            List of Comlink objects
        xgrid : array
            2D matrix of x locations
        ygrid : array
            2D matrix of y locations
        max_dist_from_cml : float
            Maximum distance from a CML path that should be considered as
            covered. The units must be the same as for the coordinates of the
            CMLs. Hence, if lat-lon is used in decimal degrees, this unit has
            also to be used here. Note that the different scaling of lat-lon
            degrees for higher latitudes is not accounted for.

        Returns
        -------

        grid_points_covered_by_cmls : array of bool
            2D array with size of `xgrid` and `ygrid` with True values where
            the grid point is within the area considered covered.
        """

        # TODO: Add option to do this for each time step, based on the
        #       available CML in self.df_cml_R, i.e. exclusing those
        #       with NaN.

        # Build a polygon for the area "covered" by the CMLs
        # given a maximum distance from their individual paths
        cml_lines = []
        for cml in cml_list:
            cml_lines.append(
                sh.geometry.LineString([
                    [cml.metadata['site_a_longitude'],
                     cml.metadata['site_a_latitude']],
                    [cml.metadata['site_b_longitude'],
                     cml.metadata['site_b_latitude']]])
                .buffer(max_dist_from_cml, cap_style=1))

        cml_dil_union = sh.ops.cascaded_union(cml_lines)
        # Build a geopandas object for this polygon
        gdf_cml_area = geopandas.GeoDataFrame(
            geometry=geopandas.GeoSeries(cml_dil_union))

        # Generate a geopandas object for all grid points
        sh_grid_point_list = [sh.geometry.Point(xy) for xy
                              in zip(xgrid.flatten(),
                                     ygrid.flatten())]
        gdf_grid_points = geopandas.GeoDataFrame(
            geometry=sh_grid_point_list)

        # Find all grid points within the area covered by the CMLs
        points_in_cml_area = geopandas.sjoin(gdf_grid_points,
                                             gdf_cml_area,
                                             how='left')

        # Generate a Boolean grid with shape of xgrid (and ygrid)
        # indicating which grid points are within the area covered by CMLs
        grid_points_covered_by_cmls = (
            (~points_in_cml_area.index_right.isnull())
            .values.reshape(xgrid.shape))

        grid_points_covered_by_cmls = grid_points_covered_by_cmls

        return grid_points_covered_by_cmls
