################################################
# Functions to calculate spatial interpolation #
################################################


import math
import numpy as np
from scipy.spatial.distance import cdist

from . import ok

def inv_dist(sample_points, sample_values, grid,power,smoothing, nn):
    """Calculate Inverse Distance Weighting Interpolation 
    
    Parameters
    ----------
    sample_points : iterable of floats
                    Locations of sample points (Lon/Lat)
                    
                    Example
                    -------
                    >>> x = np.random.rand(number_of_points)
                        y = np.random.rand(number_of_points)
                        sample_points = np.vstack((x,y)).T
                    
    sample_values : iterable of floats
                    Values at sample_points
    grid : iterable of floats
            Gridpoint locations 
            
                    Example
                    -------
                    >>> xcoords = np.arange(xstart, xstop, dx)
                        ycoords = np.arange(ystart, ystop, dy)
                        xgrid, ygrid = np.meshgrid(xcoords, ycoords)
                        xi, yi = xgrid.flatten(), ygrid.flatten()
                        grid = np.vstack((xi, yi)).T
    
    power : flt
               Power of distance decay for IDW interpolation. 
    smoothing : flt
               Power of smoothing factor for IDW interpolation. 
    nn : int 
                Number of neighbors considered for IDW interpolation
                
    """
    dist = np.sqrt(cdist(sample_points, grid)**2. + smoothing**2.)
    dist_trans = dist.T
    
    points_sorted = dist_trans.argsort(axis=1)
    
    points_used_nn = points_sorted.T[:nn].T
    values_points_used_nn = np.array(map(lambda indices: sample_values[indices],
                                         points_used_nn))
    distances_points_used_nn = np.array(map(np.take, dist_trans, points_used_nn))
    weights = 1.0 / distances_points_used_nn**power
    normalize = lambda row: row / row.sum()
    weights = np.array(map(normalize, weights))
    weight_products = weights * values_points_used_nn
    interpolated_values = weight_products.sum(axis=1)
    return interpolated_values
                          

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
                             
                                        

def kriging(sample_points, sample_values, grid,
            n_closest_points,**kwargs):
    """Calculate Ordinary Kriging interpolation 
    
    Parameters
    ----------
    sample_points : iterable of floats
                    Locations of sample points (Lon/Lat)
                    
                    Example
                    -------
                    >>> x = np.random.rand(number_of_points)
                        y = np.random.rand(number_of_points)
                        sample_points = np.vstack((x,y)).T
                    
    sample_values : iterable of floats
                    Values at sample_points
    grid : iterable of floats
            Gridpoint locations 
            
                    Example
                    -------
                    >>> xcoords = np.arange(xstart, xstop, dx)
                        ycoords = np.arange(ystart, ystop, dy)
                        xgrid, ygrid = np.meshgrid(xcoords, ycoords)
                        xi, yi = xgrid.flatten(), ygrid.flatten()
                        grid = np.vstack((xi, yi)).T     
    n_closest_points : int
                Parameters for Kriging interpolation. See OrdinaryKriging
                documentation for information.  
    kwargs : kriging parameters, optional
                See https://github.com/bsmurphy/PyKrige for details              

    Returns
    -------
    array of floats
        Interpolated values at grid points 
  
    """                 
                
    try:                
        OK = ok.OrdinaryKriging(sample_points[:,0], sample_points[:,1], sample_values,
                                verbose=False, enable_plotting=False,**kwargs)
        z, s_kr = OK.execute(style='points', xpoints=grid[:,0], ypoints=grid[:,1],
                             backend='loop',n_closest_points=n_closest_points)  
    except ValueError:
        pass
        z = np.empty(len(grid))
        
    return z    
                           
     
                          
    
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
    radius = 6371 # km
 
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
 
    return d                             
