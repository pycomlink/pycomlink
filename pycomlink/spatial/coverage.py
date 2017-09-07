import numpy as np
import shapely as sh


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

        # Get coverage for each grid point
        covered_list = []
        for i, (x_i, y_i) in enumerate(zip(xgrid.ravel(), ygrid.ravel())):
            grid_point = sh.geometry.Point((x_i, y_i))
            for cml_line in cml_lines:
                if grid_point.intersects(cml_line):
                    covered_list.append(True)
                    break
            else:
                covered_list.append(False)

        grid_points_covered_by_cmls = (np.array(covered_list)
                                       .reshape(xgrid.shape))

        return grid_points_covered_by_cmls
