#----------------------------------------------------------------------------
# Name:         comlinkset
# Purpose:      Commercial MW link Class to handle all processing steps for 
#                   a set of links
#
# Authors:      Christian Chwala, Felix Keis
#
# Created:      25.02.2015
# Copyright:    (c) Christian Chwala, Felix Keis 2015
# Licence:      The MIT License
#----------------------------------------------------------------------------

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import math
import matplotlib
import matplotlib.animation as animation

from . import wet_dry
from . import baseline
from . import A_R_relation
from . import wet_antenna
from . import mapping


class ComlinkSet():
    """
    Commercial microwave link set (CMLS) class 
        - for all data processing 
        - for multiple links
        - contains a list of Comlink Objects defined by Comlink()   
    
    Attributes
    ----------
    cml : list of pycomlink class Comlink

    
    """    
    def __init__(self, cml_list):
        self.set = cml_list
        
    def info(self):
        """
        Print ComlinkSet info 
        
        """        
        
        print '============================================================='
        print 'Number of Comlink Objects in ComlinkSet: ' + str(len(self.set))
        print 'IDs: ' 
        for cml in self.set:
            if not cml.data.empty:
                print '     ' + str(cml.metadata['link_id'])
        print '============================================================='
        

        
    def do_wet_dry_classification(self, method='std_dev', 
                                        window_length=128,
                                        threshold=1,
                                        dry_window_length=600,
                                        f_divide=1e-3,
                                        reuse_last_Pxx=False,
                                        print_info=False):
        """
        Perform wet/dry classification for all CML time series in CMLS
        
     
        Attributes:
        ---------------------------------------------
        method: str
            WIP: currently two methods are supported:
                - std_dev: Rolling standard deviation method (Schleiss & Berne, 2010)
                - stft: Rolling Fourier-transform method (Chwala et al, 2012)
        window_length: int
            length of the sliding window        
        threshold: int
            threshold which has to be surpassed to classifiy a period as 'wet'
        .....................
        Only for method stft:
        
        dry_window_length: int
            length of window for identifying dry period used for calibration purpose
        f_divide: flt
        
        reuse_last_Pxx: bool
        
        
        .....................
        print_info: bool
            print information about executed method
        -------------------------------------------    
            
        """
        for cml in self.set:
            if not cml.data.empty:
                cml.do_wet_dry_classification(method, 
                                              window_length,
                                              threshold,
                                              dry_window_length,
                                              f_divide,
                                              reuse_last_Pxx,
                                              print_info)
            
    
                

    def do_baseline_determination(self, 
                                  method='constant',
                                  wet_external=None,
                                  print_info=False):
                                      
        """
        Perform baseline determination for all CML time series in CMLS
        
        Attributes:
        ---------------------------------------------
        method: str
            supported methods:
                constant
                linear
        wet_external:      
        
        print_info: bool
            print information about executed method        
        """                              

                              
        for cml in self.set:   
            if not cml.data.empty:       
                cml.do_baseline_determination(method,
                                              wet_external,
                                              print_info)
  


    def do_wet_antenna_baseline_adjust(self,
                                       waa_max, 
                                       delta_t, 
                                       tau,
                                       wet_external=None):
                                           
        """
        Perform baseline adjustion due to wet antenna for all CML time series in CMLS
        
        Attributes:
        ---------------------------------------------
        waa_max: 

        delta_t:    
        
        tau:
        
        wet_external:
               
        """ 
                                           
        for cml in self.set:    
            if not cml.data.empty:        
                cml.do_wet_antenna_baseline_adjust(waa_max,
                                                   delta_t,
                                                   tau,
                                                   wet_external)
                
 
    def calc_A(self, remove_negative_A=True):
        
        """
        Perform calculation of attenuation for all CML time series in CMLS
        
        Attributes:
        ---------------------------------------------
        remove_negative_A: bool
            assignment: negative values of Attenuation = 0
               
        """         
        
        for cml in self.set:
            if not cml.data.empty:
                cml.calc_A(remove_negative_A)
 
                    
                    
    def calc_R_from_A(self, a=None, b=None, approx_type='ITU'):
        
        """
        Perform calculation of rain rate from attenuation for all CML time series in CMLS
        
        Attributes:
        ---------------------------------------------
        a: flt
            Parameter of A-R relationship
            If not given: Approximation considering polarization and frequency of link
        b: flt
            Parameter of A-R relationship
            If not given: Approximation considering polarization and frequency of link        
        approx_type: str
            Type used for approximate a and b
            Supported type : ITU
            
               
        """          
        for cml in self.set:
            if not cml.data.empty:
                cml.calc_R_from_A(a,b,approx_type)                
                
                                                           
        
            
    def plot_idw(self, area, grid_res,
                 figsize=(15,10),
                 acc_type='sum',
                 time_resolution=15,
                 power=2,smoothing=0):
        """
        Plotting Inverse Distance Interpolation of Rain Sums or Rain Rate 
            on regular grid
        
        Parameter:
        ---------------
        area: list
            [lower_lon,upper_lon,lower_lat,upper_lat]
            
        grid_res: int
            number of bins of output grid in area
            
        figsize:
            size of output figure    
            
        acc_type: str
            type of accumulation
                'sum' precipitation sum of selected period
                'rr' rain rate resampled by time_resolution
        time_resolution: int
                resampling time for rain rate calculation
                only used for type 'rr'
                
        power, smoothing: flt
                 power of distance decay and smoothing factor for 
                 IDW interpolation       
        """
        
        fig = plt.figure(figsize=figsize)
        mp = Basemap(projection='merc',llcrnrlat=area[2],urcrnrlat=area[3],\
            llcrnrlon=area[0],urcrnrlon=area[1],lat_ts=20,resolution=None)
        mp.shadedrelief() 
        # draw parallels.
        parallels = np.arange(40.,60.,0.5)
        mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
        # draw meridians
        meridians = np.arange(0.,20.,0.5)
        mp.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10) 
        
        
        nws_precip_colors = [
        "#04e9e7",  # 0.01 - 0.10 mm
        "#019ff4",  # 0.10 - 0.25 mm
        "#0300f4",  # 0.25 - 0.50 m
        "#02fd02",  # 0.50 - 0.75
        "#01c501",  # 0.75 - 1.00 
        "#008e00",  # 1.00 - 1.50 
        "#fdf802",  # 1.50 - 2.00 
        "#e5bc00",  # 2.00 - 2.50 
        "#fd9500",  # 2.50 - 3.00 
        "#fd0000",  # 3.00 - 4.00
        "#d40000",  # 4.00 - 5.00 
        "#bc0000",  # 5.00 - 6.00 
        "#f800fd",  # 6.00 - 8.00 
        "#9854c6",  # 8.00 - 10.00 
        "#fdfdfd"   # 10.00+
        ]
        precip_colormap = matplotlib.colors.ListedColormap(nws_precip_colors)        
        levels_rr = [0.01, 0.1, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
          6.0, 8.0, 10., 100.0]  
        levels_sum = [1.00, 2.0, 3.0, 4.0, 5.0,6.0, 7.0,8.0,9.0, 10.0, 15.0, 20.0,
          25.0, 50.0, 100.,500.]           
        norm_rr = matplotlib.colors.BoundaryNorm(levels_rr, 15)  
        norm_sum = matplotlib.colors.BoundaryNorm(levels_sum, 15)
        
        #Definition of output grid
        gridx = np.linspace(area[0],area[1],grid_res)
        gridy = np.linspace(area[2],area[3],grid_res)
        grid = np.meshgrid(gridx,gridy)    
        x,y = mp(grid[0],grid[1])

       
        # MW data and metadata
        lons_mw=[]
        lats_mw=[]
        values_mw=[]
        if acc_type == 'sum':
            for cml in self.set: 
                if not cml.data.empty:
                   prep_sum=((cml.data.R_fn.resample('H',how='mean')+
                                       cml.data.R_nf.resample('H',how='mean'))/2.).sum() 
                   if not math.isnan(prep_sum):                     
                       lons_mw.append((cml.metadata['site_A']['lon']
                                      +cml.metadata['site_B']['lon'])/2.)
                       lats_mw.append((cml.metadata['site_A']['lat']
                                      +cml.metadata['site_B']['lat'])/2.)
                       values_mw.append(prep_sum)
                                   
            inv_d_values=mapping.inv_dist(lons_mw,lats_mw,values_mw,
                                          gridx,gridy,power,smoothing)
            
                                                                
            cs = mp.contourf(x,y,inv_d_values,levels=levels_sum,norm=norm_sum,cmap=precip_colormap)
            ln,lt = mp(lons_mw,lats_mw)

            mp.scatter(ln,lt,c=values_mw,cmap=precip_colormap, alpha=0.6, s=60,norm=norm_sum)            
            cbar = mp.colorbar(cs,location='bottom',pad="5%")
            cbar.set_label('mm')
  
        elif acc_type == 'rr':

            def animate(i):

                for cml in self.set:
                    if not cml.data.empty:

                        prep_rr = (cml.data.R_fn.resample(str(time_resolution)+'Min',how='mean')[i]+
                                   cml.data.R_nf.resample(str(time_resolution)+'Min',how='mean')[i])/2.                
                    
                
                        if not math.isnan(prep_rr):                     
                            lons_mw.append((cml.metadata['site_A']['lon']
                                           +cml.metadata['site_B']['lon'])/2.)
                            lats_mw.append((cml.metadata['site_A']['lat']
                                           +cml.metadata['site_B']['lat'])/2.)
                            values_mw.append(prep_rr)   
    
                inv_d_values=mapping.inv_dist(lons_mw,lats_mw,values_mw,
                                              gridx,gridy,
                                              power,smoothing)                           
                
                plt.title(str(self.set[0].data.resample(str(time_resolution)+'Min').index[i]))
    
                cs = mp.contourf(x,y,inv_d_values,levels=levels_rr,norm=norm_rr,cmap=precip_colormap)
                ln,lt = mp(lons_mw,lats_mw)
                mp.scatter(ln,lt,c=values_mw,cmap=precip_colormap, alpha=0.6, s=60,norm=norm_rr)                
                cbar = mp.colorbar(cs,location='bottom',pad="5%")
                cbar.set_label('mm/h')
                            
            anim = animation.FuncAnimation(fig, animate, 
                                           frames=len(self.set[0].data.resample(str(time_resolution)+'Min').index),
                                           interval=1000, blit=True)        
                            
            anim.save('animation.gif', writer='imagemagick')    
                         
            
        else:
            ValueError('acc_type has to be "sum" or "rr"');
        


                 