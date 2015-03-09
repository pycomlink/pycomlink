#----------------------------------------------------------------------------
# Name:         comlinkset
# Purpose:      Commercial MW link Class to handle all processing steps for 
#                   a set of links
#
# Authors:      Felix Keis
#
# Created:      25.02.2015
# Copyright:    (c) Felix Keis 2014
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
    Commercial microwave link (CML) class for all data processing 
    for multiple links
    
    contains a list     
    
    Attributes
    ----------
    cml : list of pycomlink class Comlink

    
    """    
    def __init__(self, cml_list):
        self.set = cml_list
        
    def info(self):
        """
        Print comlinkSet info 
        
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
        """Perform wet/dry classification for CML time series
        
        WIP: Currently  two methods are supported:
                -std_dev
                -stft
                
        TODO: Docstring!!!
        
        """
        for cml in self.set:
            if not cml.data.empty:
            
                # Standard deviation method
                if method == 'std_dev':
                    if print_info:
                        print 'Performing wet/dry classification'
                        print ' Method = std_dev'
                        print ' window_length = ' + str(window_length)
                        print ' threshold = ' + str(threshold)
                    for pair_id in cml.processing_info['tx_rx_pairs']:
                        (cml.data['wet_' + pair_id], 
                         roll_std_dev) = wet_dry.wet_dry_std_dev(
                                            cml.data['txrx_' + pair_id].values, 
                                            window_length, 
                                            threshold)
                        cml.processing_info['wet_dry_roll_std_dev_' + pair_id] \
                                      = roll_std_dev
                    cml.processing_info['wet_dry_method'] = 'std_dev'
                    cml.processing_info['wet_dry_window_length'] = window_length
                    cml.processing_info['wet_dry_threshold'] = threshold
                # Shor-term Fourier transformation method
                elif method == 'stft':
                    if print_info:
                        print 'Performing wet/dry classification'
                        print ' Method = stft'
                        print ' dry_window_length = ' + str(dry_window_length)
                        print ' window_length = ' + str(window_length)
                        print ' threshold = ' + str(threshold)
                        print ' f_divide = ' + str(f_divide)
                
                    for pair_id in cml.processing_info['tx_rx_pairs']:
                        txrx = cml.data['txrx_' + pair_id].values
                
                        # Find dry period (wit lowest fluctuation = lowest std_dev)
                        t_dry_start, \
                        t_dry_stop = wet_dry.find_lowest_std_dev_period(
                                        txrx,
                                        window_length=dry_window_length)
                        cml.processing_info['wet_dry_t_dry_start'] = t_dry_start
                        cml.processing_info['wet_dry_t_dry_stop'] = t_dry_stop
                
                        if reuse_last_Pxx is False:
                            cml.data['wet_' + pair_id], info = wet_dry.wet_dry_stft(
                                                                    txrx,
                                                                    window_length,
                                                                    threshold,
                                                                    f_divide,
                                                                    t_dry_start,
                                                                    t_dry_stop)
                        elif reuse_last_Pxx is True:
                            Pxx=cml.processing_info['wet_dry_Pxx_' + pair_id]
                            f=cml.processing_info['wet_dry_f']
                            cml.data['wet_' + pair_id], info = wet_dry.wet_dry_stft(
                                                                    txrx,
                                                                    window_length,
                                                                    threshold,
                                                                    f_divide,
                                                                    t_dry_start,
                                                                    t_dry_stop,
                                                                    Pxx=Pxx,
                                                                    f=f)
                        else:
                            raise ValueError('reuse_last_Pxx can only by True or False')
                        cml.processing_info['wet_dry_Pxx_' + pair_id] = \
                                                        info['Pxx']
                        cml.processing_info['wet_dry_P_norm_' + pair_id] = \
                                                        info['P_norm']
                        cml.processing_info['wet_dry_P_sum_diff_' + pair_id] = \
                                                        info['P_sum_diff']
                        cml.processing_info['wet_dry_P_dry_mean_' + pair_id] = \
                                                        info['P_dry_mean']
                
                    cml.processing_info['wet_dry_f'] = info['f']                
                    cml.processing_info['wet_dry_method'] = 'stft'
                    cml.processing_info['wet_dry_window_length'] = window_length
                    cml.processing_info['dry_window_length'] = window_length
                    cml.processing_info['f_divide'] = f_divide
                    cml.processing_info['wet_dry_threshold'] = threshold
                else:
                    ValueError('Wet/dry classification method not supported')     
                

    def do_baseline_determination(self, 
                                  method='constant',
                                  wet_external=None,
                                  print_info=False):
        for cml in self.set:   
            if not cml.data.empty:                           
                if method == 'constant':
                    baseline_func = baseline.baseline_constant
                    if print_info:
                        print 'Setting RSL baseline'
                        print ' Method = constant'
                elif method == 'linear':
                    baseline_func = baseline.baseline_linear
                    if print_info:
                        print 'Setting RSL baseline'
                        print ' Method = linear'
                else:
                    ValueError('Wrong baseline method')
                for pair_id in cml.processing_info['tx_rx_pairs']:
                    if wet_external is None:            
                        wet = cml.data['wet_' + pair_id]
                    else:
                        wet = wet_external
                    cml.data['baseline_' + pair_id] = \
                                    baseline_func(cml.data['txrx_' + pair_id], 
                                                  wet)
                cml.processing_info['baseline_method'] = method    


    def do_wet_antenna_baseline_adjust(self,
                                       waa_max, 
                                       delta_t, 
                                       tau,
                                       wet_external=None):
        for cml in self.set:    
            if not cml.data.empty:                               
                for pair_id in cml.processing_info['tx_rx_pairs']:
                    txrx = cml.data['txrx_' + pair_id].values
                    baseline = cml.data['baseline_' + pair_id].values
                    if wet_external is None:            
                        wet = cml.data['wet_' + pair_id]
                    else:
                        wet = wet_external
                    baseline_waa, waa = wet_antenna.waa_adjust_baseline(rsl=txrx,
                                                               baseline=baseline,
                                                               waa_max=waa_max,
                                                               delta_t=delta_t,
                                                               tau=tau,
                                                               wet=wet)
                    cml.data['baseline_' + pair_id] = baseline_waa
                    cml.data['waa_' + pair_id] = waa 
                
 
    def calc_A(self, remove_negative_A=True):
        for cml in self.set:
            if not cml.data.empty:

                for pair_id in cml.processing_info['tx_rx_pairs']:
                    cml.data['A_' + pair_id] = cml.data['txrx_' + pair_id] \
                                              - cml.data['baseline_' + pair_id]
                    if remove_negative_A:
                        cml.data['A_' + pair_id][cml.data['A_' + pair_id]<0] = 0  
                    
                    
    def calc_R_from_A(self, a=None, b=None, approx_type='ITU'):
        for cml in self.set:
            if not cml.data.empty:
                if a==None or b==None:
                    calc_a_b = True
                else:
                    calc_a_b = False
                for pair_id in cml.processing_info['tx_rx_pairs']:
                    if calc_a_b:
#                        a, b = A_R_relation.a_b(f_GHz=cml.metadata['f_GHz_' \
#                                                                    + pair_id], 
#                                                pol=cml.metadata['pol_' \
#                                                                  + pair_id],
#                                                approx_type=approx_type)
                                                
                        a, b = A_R_relation.a_b(f_GHz=cml.tx_rx_pairs[pair_id]['f_GHz'], 
                                        pol=cml.tx_rx_pairs[pair_id]['pol'],
                                        approx_type=approx_type)
                        cml.processing_info['a_' + pair_id] = a
                        cml.processing_info['b_' + pair_id] = b
    
                    cml.data['R_' + pair_id] = \
                        A_R_relation.calc_R_from_A(cml.data['A_' + pair_id], 
                                                   a, b,
                                                   cml.metadata['length_km'])
                                                              
        
            
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

        #mp.drawcoastlines(color='blue')
        #mp.drawcountries()
        #mp.etopo()
        #mp.drawrivers(color='blue')         
        
        
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
            
            #cs = mp.pcolormesh(x,y,inv_d_values, norm=norm,
            #               cmap=precip_colormap)
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
        


                 