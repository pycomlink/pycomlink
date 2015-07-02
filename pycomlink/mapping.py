################################################
# Functions to calculate spatial interpolation #
################################################

from math import sqrt
import numpy as np

def inv_dist(lons_mp,lats_mp,values_mp,lons_grid,lats_grid,power,smoothing):  
    """Calculate Inverse Distance Weighting interpolation values at regular grid
    
    Parameters
    ----------
    lons_mp : iterable of floats
               Longitudes of measuring points
    lats_mp : iterable of floats
               Latitudes of measuring points               
    values_mp : iterable of floats
               Values at measuring points    
    lons_grid : array of floats
               Longitudes of regular grid 
    lats_grid : array of floats
               Latitudes of regular grid   
    power : flt
               Power of distance decay for IDW interpolation. 
    smoothing : flt
               Power of smoothing factor for IDW interpolation. 
               
    Returns
    -------
    array of floats
         interpolated values at grid points           
               
    """        
    
    values_idw = np.zeros((lats_grid.size,lons_grid.size))  
    
    for i in range(0,lons_grid.size):
        for j in range(0,lats_grid.size):
            values_idw[j][i] = gridpointValue(lons_grid[i],lats_grid[j],power,smoothing,
                                                lons_mp,lats_mp,values_mp) 
    return values_idw


def gridpointValue(x,y,power,smoothing,x_o,y_o,values_o):  
    """Calculate IDW value at particular grid point
    
    Parameters
    ----------
    x : float
        Longitude of grid point
    y : float
        Latitude of grid point
    power : flt
        Power of distance decay
    smoothing : flt
        Power of smoothing factor
    x_o : iterable of floats
        Longitudes of measuring points
    y_o : iterable of floats
        Latitudes of measuring points               
    values_o : iterable of floats
         Values at measuring points        

    Returns
    -------
    float
        Value at grid point 
    
    """
    
    nominator=0  
    denominator=0 

    for ii in range(0,len(values_o)):  
        # WIP to be revised        
        #very simple distance calculation
        dist = sqrt((x-x_o[ii])*(x-x_o[ii])+(y-y_o[ii])*(y-y_o[ii])+smoothing*smoothing)
        #If the point is really close to one of the data points, 
        #            return the data point value to avoid singularities  
        if(dist<0.0000000001):  
            return values_o[ii]  
        nominator=nominator+(values_o[ii]/pow(dist,power))  
        denominator=denominator+(1/pow(dist,power))  
    #Return NODATA if the denominator is zero  
    if denominator > 0:  
        value = nominator/denominator  
    else:  
        value = -9999  
    return value
                           

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
                             
                             
def kriging(lons_mp,lats_mp,values_mp,lons_grid,lats_grid,
            krig_type,variogram_model,drift_terms):
                
    """Calculate Kriging interpolation values at regular grid
    
    Parameters
    ----------
    lons_mp : iterable of floats
               Longitudes of measuring points
    lats_mp : iterable of floats
               Latitudes of measuring points               
    values_mp : iterable of floats
               Values at measuring points    
    lons_grid : array of floats
               Longitudes of regular grid 
    lats_grid : array of floats
               Latitudes of regular grid    
    krig_type : str
                Parameters for Kriging interpolation. See pykrige documentation
                for information.
    variogram_model : str
                Parameters for Kriging interpolation. See pykrige documentation
                for information.       
    drift_terms : str
                Parameters for Kriging interpolation. See pykrige documentation
                for information.              

    Returns
    -------
    array of floats
        Interpolated values at grid points 
  
    """     
    
    from pykrige.ok import OrdinaryKriging
    from pykrige.uk import UniversalKriging
    
    if krig_type == 'ordinary':
       OK = OrdinaryKriging(lons_mp, lats_mp, values_mp,
                            variogram_model=variogram_model,
                            verbose=False, enable_plotting=False) 
       z, s_kr = OK.execute('grid', lons_grid, lats_grid)   
    elif krig_type == 'universal':
       UK = UniversalKriging(lons_mp, lats_mp, values_mp,
                            variogram_model=variogram_model,
                            drift_terms=drift_terms,
                            verbose=False, enable_plotting=False) 
       z, s_kr = UK.execute('grid', lons_grid, lats_grid) 
    else:
       ValueError('Kriging type not supported')   
    return z       
                          
    
                             