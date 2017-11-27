#----------------------------------------------------------------------------
# Name:         
# Purpose:      
#
# Authors:     Christian Chwala, Felix Keis
#
# Created:      
# Copyright:    (c) Christian Chwala 2016
# Licence:      The MIT License
#----------------------------------------------------------------------------

from __future__ import division
from builtins import map
from math import radians, cos, sin, asin, sqrt


# TODO: Check if these functions are still needed

def distance(origin, destination):
    """Simple distance (in km) calculation between two locations

    Parameters
    ----------
    origin : tuple
        Coordinates of first location in decimal format.
        Required format (Latitude,Longitude)
    destination : tuple
        Coordinates of second location in decimal format.
        Required format (Latitude,Longitude)

    Returns
    -------
    Distance between origin and destination in kilometer

    """

    lat1, lon1 = origin
    lat2, lon2 = destination

    return haversine(lon1, lat1, lon2, lat2)


def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = list(map(radians, [lon1, lat1, lon2, lat2]))
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km


def label_loc(lon_a,lat_a,lon_b,lat_b):
    """Helper function for method info_plot of class Comlink
    """
    if lon_a < lon_b and lat_a < lat_b:
        x_a,y_a = lon_a-0.025,lat_a-0.005
        x_b,y_b = lon_b+0.01,lat_b+0.005
    elif lon_a < lon_b and lat_a > lat_b:
        x_a,y_a = lon_a-0.025,lat_a+0.005
        x_b,y_b = lon_b+0.01,lat_b-0.005
    elif lon_a > lon_b and lat_a > lat_b:
        x_a,y_a = lon_a+0.01,lat_a+0.005
        x_b,y_b = lon_b-0.025,lat_b-0.005
    elif lon_a > lon_b and lat_a < lat_b:
        x_a,y_a = lon_a+0.01,lat_a-0.005
        x_b,y_b = lon_b-0.025,lat_b+0.005
    xy = [x_a,y_a,x_b,y_b]
    return xy