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
import folium


def plot_cml_paths(cml_list, fol_map=None, tiles='OpenStreetMap', **kwargs):
    """

    @param cml_list:
    @param fol_map:
    @param tiles:
    @return:
    """

    lats = []
    lons = []
    for cml in cml_list:
        coords = cml.get_coordinates()

        if fol_map is None:
            fol_map = folium.Map(location=[(coords.lat_a + coords.lat_b) / 2,
                                           (coords.lon_a + coords.lon_b) / 2],
                                 tiles=tiles,
                                 zoom_start=8)
        fol_map.add_child(folium.PolyLine([(coords.lat_a, coords.lon_a),
                                              (coords.lat_b, coords.lon_b)],
                                             **kwargs))
        lats.append(coords.lat_a)
        lats.append(coords.lat_b)
        lons.append(coords.lon_a)
        lons.append(coords.lon_b)

    fol_map.location = [(max(lats) + min(lats)) / 2,
                        (max(lons) + min(lons)) / 2]

    return fol_map
