################################################
# Functions to calculate spatial interpolation #
################################################

from math import sqrt
import numpy as np

def inv_dist(lons_mp,lats_mp,values_mp,lons_grid,lats_grid,power,smoothing):  
    """
    lons_mp,lats_mp coordinates of measuring points
    values_mp values at measuring points
    
    lons_grid,lats_grid coordinates of regular grid    
    """        
    
    values_idw = np.zeros((lats_grid.size,lons_grid.size))  
    
    for i in range(0,lons_grid.size):
        for j in range(0,lats_grid.size):
            values_idw[j][i] = gridpointValue(lons_grid[i],lats_grid[j],power,smoothing,
                                                lons_mp,lats_mp,values_mp) 
    return values_idw


def gridpointValue(x,y,power,smoothing,x_o,y_o,values_o):  
    """
    x,y coordinates lon/lat of regular grid
    x_o,y_o coordinates lon,lat of measuring point
    values_o value at measuring point    
    
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
                           
                             