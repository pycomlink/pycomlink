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
import cartopy
from cartopy.io import img_tiles
import pandas as pd

import math
import matplotlib

from . import wet_dry
from . import baseline
from . import A_R_relation
from . import wet_antenna
from . import mapping


class ComlinkSet():
    """Commercial microwave link set (CMLS) class for data processing of multiple links
    
    Attributes
    ----------
    set : list of pycomlink class Comlink objects
    set_info : dict
          Dictionary that holds information about the CMLS object
        
    """  
    
    def __init__(self, cml_list,area):
        """Initialisation of commercial microwave link set (CMLS) class 
           
            Parameters
            ----------
            cml_list : list of pycomlink class Comlink objects
                List that holds the links located in a specified area. Each list item
                has to be a Comlink object.
            area : list of floats
                List that holds the geographic coordinates of the area borders in
                decimal format. 
                The order is mandatory.
                
                Example
                -------
                >>> area = [lower_longitude,upper_longitude,lower_latitude, upper_latitude]
                
        """
        
        
        self.set = cml_list
        self.set_info = {'area':area}
        
    def info(self):
        """Print information about associated microwave links 
        
        """        
        
        print '============================================================='
        print 'Number of Comlink Objects in ComlinkSet: ' + str(len(self.set))
        print '  ----- '+"{:.2f}".format(self.set_info['area'][3])+' -----'
        print '  |               |'
        print "{:.2f}".format(self.set_info['area'][0])+'   area    ' \
                +"{:.2f}".format(self.set_info['area'][1])
        print '  |               |'        
        print '  ----- '+"{:.2f}".format(self.set_info['area'][2])+' -----'        

        print 'IDs: ' 
        for cml in self.set:
            print '     ' + str(cml.metadata['link_id'])
        print '============================================================='
        

    def info_plot(self,out_file=None):
        """Plot associated links on a map 
                
        """
        plt.figure(figsize=(10,10))
        ax = plt.axes(projection=cartopy.crs.PlateCarree())

        lons=[]
        lats=[]
        for cml in self.set:
            lons.append(cml.metadata['site_A']['lon'])
            lons.append(cml.metadata['site_B']['lon'])
            lats.append(cml.metadata['site_A']['lat'])
            lats.append(cml.metadata['site_B']['lat'])   
            
        area=[min(lons)-.05,
              max(lons)+.05,
              min(lats)-.05,
              max(lats)+.05]           
        
        ax.set_extent((area[0], area[1], area[2], area[3]), crs=cartopy.crs.PlateCarree())
        gg_tiles=img_tiles.MapQuestOSM()
        ax.add_image(gg_tiles, 11)        
        
        for cml in self.set:
                   plt.plot([cml.metadata['site_A']['lon'],cml.metadata['site_B']['lon']],
                            [cml.metadata['site_A']['lat'],cml.metadata['site_B']['lat']],
                            linewidth=2,color='k',
                            transform=cartopy.crs.Geodetic())        
        
        if out_file is not None:
            plt.savefig(out_file,format='png',bbox_inches='tight', dpi=300) 


    def quality_test(self,rx_range=[-80,-10],tx_range=[-6,35],figsize=(6,4)):
        """Perform quality tests of TX, RX time series and print information
        
        Parameters
        ----------
        rx_range : list, optional
            List of lower and upper limit of plausible RX values in dBm. 
            Default is [-75,-10]
        tx_range : list, optional
            List of lower and upper limit of plausible TX values in dBm. 
            Default is [-6,35]    
        figsize : matplotlib parameter, optional 
            Size of output figure in inches (default is (6,4))    
                   
        """
        for cml in self.set:
            cml.quality_test(rx_range,tx_range,figsize)


    def remove_bad_values(self,bad_value=-99.9):
        """Detect bad values and convert to NaN
        
        Parameters
        ----------
        bad_value : int or float
                Bad value to be removed
        
        """
        for cml in self.set:
            cml.remove_bad_values(bad_value)

    
    def do_wet_dry_classification(self, method='std_dev', 
                                        window_length=128,
                                        threshold=1,
                                        dry_window_length=600,
                                        f_divide=1e-3,
                                        reuse_last_Pxx=False,
                                        print_info=False):
        """Perform wet/dry classification for all CML time series in CMLS

        Parameters
        ----------
        method : str, optional
                 String which indicates the classification method (see Notes)
                 Default is 'std_dev'
        window_length : int, optional
                 Length of the sliding window (Default is 128)        
        threshold : int, optional
                 threshold which has to be surpassed to classifiy a period as 'wet'
                 (Default is 1)        
        dry_window_length : int, optional
                 Length of window for identifying dry period used for calibration
                 purpose. Only for method stft. (Default is 600)
        f_divide : float
                 Parameter for classification with method Fourier transformation
                 (Default is 1e-3)          
        reuse_last_Pxx : bool
                 Parameter for classification with method Fourier transformation
                 (Default is false)  
        print_info : bool
                  Print information about executed method (Default is False)
        
        Note
        ----        
        WIP: Currently two classification methods are supported:
                - std_dev: Rolling standard deviation method [1]_
                - stft: Rolling Fourier-transform method [2]_                
                      
        
        """
        
        for cml in self.set:
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
                                      
        """Perform baseline determination for all CML time series in CMLS
        
        Parameters
        ----------
        method : str, optional
                 String which indicates the baseline determination method (see Notes)
                 Default is 'constant'
        wet_external : iterable of int or iterable of float, optional
                 External wet/dry classification information. Has to be specified
                 for each classified index of times series (Default is None)        
        print_info : bool
                 Print information about executed method (Default is False)
                 
        Note
        ----
        WIP: Currently two methods are supported:
             - constant: Keeping the RSL level constant at the level of the preceding dry period
             - linear: interpolating the RSL level linearly between two enframing dry periods
             
        """                            

                              
        for cml in self.set:   
            cml.do_baseline_determination(method,
                                              wet_external,
                                              print_info)
  


    def do_wet_antenna_baseline_adjust(self,
                                       waa_max, 
                                       delta_t, 
                                       tau,
                                       wet_external=None):
                                           
        """Perform baseline adjustion due to wet antenna for all CML time series in CMLS
        
        Parameters
        ----------
        waa_max : float
                  Maximum value of wet antenna attenuation   
        delta_t : float
                  Parameter for wet antnenna attenation model    
        tau : float
              Parameter for wet antnenna attenation model 
        
        wet_external : iterable of int or iterable of float, optional
                 External wet/dry classification information. Has to be specified
                 for each classified index of times series (Default is None)
        
        Note
        ----        
        The wet antenna adjusting is based on a peer-reviewed publication [3]_
        
  
        """ 
                                           
        for cml in self.set:    
            cml.do_wet_antenna_baseline_adjust(waa_max,
                                                   delta_t,
                                                   tau,
                                                   wet_external)
                
 
    def calc_A(self, remove_negative_A=True):
        
        """Perform calculation of attenuation for all CML time series in CMLS
        
        Parameters
        ----------
        remove_negative_A : bool
                Negative attenuation values are assigned as zero (Default is True)
               
        """         
        
        for cml in self.set:
            cml.calc_A(remove_negative_A)
 
                    
                    
    def calc_R_from_A(self, a=None, b=None, approx_type='ITU'):
        
        """Perform calculation of rain rate from attenuation for all CML time series in CMLS
        
        Parameters
        ----------
        a : float, optional
            Parameter of A-R relationship (Default is None)
            If not given: Approximation considering polarization and frequency of link
        b : float, optional
            Parameter of A-R relationship  (Default is None)
            If not given: Approximation considering polarization and frequency of link    
        approx_type : str, optional
            Approximation type (the default is 'ITU', which implies parameter
            approximation using a table recommanded by ITU) 
            
               
        """          
        for cml in self.set:
            for pair_id in cml.processing_info['tx_rx_pairs']:
<<<<<<< HEAD
                if cml.processing_info['tx_rx_pairs'][pair_id]['f_GHz'] is not None: 
                   cml.calc_R_from_A(a,b,approx_type)     
                else:
                   cml.data['R_'+pair_id] = None 
                
                                                           
=======
                if not cml.processing_info['tx_rx_pairs'][pair_id]['f_GHz']:
                    remove=True
            if remove:
                self.set.remove(cml)        
        
        for cml in self.set:
                cml.calc_R_from_A(a,b,approx_type)     
                                                                    
>>>>>>> 6356332... Revision of plotting and spatial interpolation methods. IDW now with limitation to nn neighbors. Plotting in spat_interpol now optional
               
    def spat_interpol(self, grid_res,
                 int_type,
                 figsize=(15,10),
                 time=None,
                 method='mean',
<<<<<<< HEAD
                 time_resolution=15,
                 power=2,smoothing=0,
=======
                 power=2,smoothing=0,nn=10,
>>>>>>> 6356332... Revision of plotting and spatial interpolation methods. IDW now with limitation to nn neighbors. Plotting in spat_interpol now optional
                 krig_type='ordinary',
                 variogram_model='linear',
                 drift_terms=['regional_linear'],
                 plot=True,
                 out_file=None):
                     
        """ Perform and plot spatial interpolation of rain rate or rain sum on regular grid
        
        Parameters
        ----------   
        grid_res : int
            Number of bins of output grid in area            
        int_type : str
            interpolation method
            'IDW' Inverse Distance Interpolation
            'Kriging' Kriging Interpolation with pyKrige            
        figsize : matplotlib parameter, optional 
            Size of output figure in inches (default is (15,10))      
        time : str, optional
            Datetime string of desired timestamp (for example 'yyyy-mm-dd HH:MM')
            Default is None.
            If given the rain rate for this timestamp is plotted.
            If not given the accumulation of the whole time series is plotted.         
        method : str, optional
            Indicates how to claculate rainrate/sum of bidirectional link
            (Default is 'mean')
            'mean' average of near-far and far-near
            'max' maximum of near-far and far-near
            'min' minimum of near-far and far-near
            'nf', 'fn', 'fnp' etc. use direction from tx_rx_pairs               
        time_resolution : int, optional
                Resampling time for rain rate calculation. Only used if time
                is not None (Default is 15)               
        power : flt, optional
               Power of distance decay for IDW interpolation. Only used if 
               int_type is 'IDW' (Default is 2)   
        smoothing : flt, optional
               Power of smoothing factor for IDW interpolation. Only used if 
               int_type is 'IDW' (Default is 0) 
        nn : int,optional
                Number of neighbors considered for IDW interpolation. Only used
                if int_type is 'IDW' (Default is 10)
        krig_type : str, optional
                Parameters for Kriging interpolation. See pykrige documentation
                for information. Only used if int_type is 'Kriging' 
                (Default is 'Ordinary')
        variogram_model : str, optional
                Parameters for Kriging interpolation. See pykrige documentation
                for information. Only used if int_type is 'Kriging' 
                (Default is 'linear')        
        drift_terms : str, optional
                Parameters for Kriging interpolation. See pykrige documentation
                for information. Only used if int_type is 'Kriging' 
                (Default is ['regional_linear'])
        plot : bool, optional
                Plot the interpolation on low resolution OSM
                (Default is True)
        out_file : str, optional
                file path of output image file
                (Default is None)
                
        """

<<<<<<< HEAD
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=cartopy.crs.PlateCarree())
        gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')        
        gl.xlabels_top = False
        ax.set_extent((self.set_info['area'][0]-.05, self.set_info['area'][1]+.05,
                       self.set_info['area'][2]-.05, self.set_info['area'][3]+.05),
                         crs=cartopy.crs.PlateCarree())
        gg_tiles = img_tiles.OSM()

        ax.add_image(gg_tiles, 11)                 
        for cml in self.set:
            if 'site_A' in cml.metadata and 'site_B' in cml.metadata:
                if 'lat' in cml.metadata['site_A'] and \
                   'lon' in cml.metadata['site_A'] and \
                   'lat' in cml.metadata['site_B'] and \
                   'lon' in cml.metadata['site_B']:
                       plt.plot([cml.metadata['site_A']['lon'],cml.metadata['site_B']['lon']],
                         [cml.metadata['site_A']['lat'],cml.metadata['site_B']['lat']],
                         linewidth=1,color='k',
                         transform=cartopy.crs.Geodetic()) 


        levels_rr = [0.5, 1.0, 1.5,2.0, 2.5, 5.0, 7.5, 10.0,12.5,15.0,20.0]
        levels_sum = [5.0,10.0,15.0,20.0,25.0,50.0,75.0,100.0,150.0,200.0,250.0]          

        
=======
>>>>>>> 6356332... Revision of plotting and spatial interpolation methods. IDW now with limitation to nn neighbors. Plotting in spat_interpol now optional
        #Definition of output grid
        gridx = np.linspace(self.set_info['area'][0],self.set_info['area'][1],grid_res)
        gridy = np.linspace(self.set_info['area'][2],self.set_info['area'][3],grid_res)   
        grid = np.meshgrid(gridx, gridy)
        
        lons_mw=[]
        lats_mw=[]
        values_mw=[]
        
        # MW data and metadata      
        for cml in self.set:
            if 'site_A' in cml.metadata and 'site_B' in cml.metadata:
                if 'lat' in cml.metadata['site_A'] and \
                   'lon' in cml.metadata['site_A'] and \
                   'lat' in cml.metadata['site_B'] and \
                   'lon' in cml.metadata['site_B']:
                         
                       lat_center = (cml.metadata['site_A']['lat']
                                       +cml.metadata['site_B']['lat'])/2.
                       lon_center = (cml.metadata['site_A']['lon']
                                       +cml.metadata['site_B']['lon'])/2.                                                           
                           
                       if time is None:                
                           plist = []
                           for pair_id in cml.processing_info['tx_rx_pairs']:
<<<<<<< HEAD
                               if 'R_' + pair_id in cml.data.columns:
                                   cml.processing_info['accR_' + pair_id] = cml.data['R_' + pair_id].resample('H',how='mean').cumsum()[-1]  

        lons_mw=[]
        lats_mw=[]
        values_mw=[]        
 
        if time is None:
             for cml in self.set:
                 plist = []
                 for pair_id in cml.processing_info['tx_rx_pairs']:
                     try:
                         plist.append(cml.processing_info['accR_'+pair_id])
                     except:
                         pass       
                 if method == 'mean':
                     try:
                         precip = np.mean(plist)
                     except ValueError:
                         pass
                     except TypeError:
                         pass                     
                 elif method == 'max':    
                     try:
                         precip = np.max(plist)
                     except ValueError:
                         pass
                     except TypeError:
                         pass                      
                 elif method == 'min':
                     try:
                         precip = np.min(plist)  
                     except ValueError:
                         pass
                     except TypeError:
                         pass                      
                 elif method in cml.processing_info['tx_rx_pairs']:
                     if 'accR_' + pair_id in cml.processing_info.keys():
                         precip = cml.processing_info['accR_'+method]
                     else:
                         print 'Pair ID '+method+' not available for link '+cml.metadata['link_id']
                         precip = None                         
                 else:
                     print method+' not available for link '+cml.metadata['link_id']
                     precip = None     

                 if precip >= 0.:                                     
                     lons_mw.append(cml.metadata['lon_center'])
                     lats_mw.append(cml.metadata['lat_center'])
                     values_mw.append(precip)     
  
              
        else:
             start = pd.Timestamp(time) - pd.Timedelta('30s')
             stop = pd.Timestamp(time) + pd.Timedelta('30s')
             for cml in self.set:
                 plist = []
                 for pair_id in cml.processing_info['tx_rx_pairs']:
                     try:
                         plist.append((cml.data['R_'+pair_id].resample(str(time_resolution)+'Min',how='mean')[start:stop]).values[0])
                     except:
                         pass
 
                 if method == 'mean':
                     try:
                         precip = np.mean(plist)
                     except ValueError:
                         pass
                     except TypeError:
                         pass                        
                 elif method == 'max':
                     try:
                         precip = np.max(plist)
                     except ValueError:
                         pass
                     except TypeError:
                         pass                        
                 elif method == 'min':
                     try:
                         precip = np.min(plist)  
                     except ValueError:
                         pass
                     except TypeError:
                         pass                        
                 elif method in cml.processing_info['tx_rx_pairs']:
                     if 'R_' + pair_id in cml.data.keys():
                         precip = (cml.data['R_'+method].resample(str(time_resolution)+'Min',how='mean')[start:stop]).values[0]
                     else:
                         print 'Pair ID '+method+' not available for link '+cml.metadata['link_id']
                         precip = None
                 else:
                     print method+' not available for link '+cml.metadata['link_id']
                     precip = None
     
                 if precip >= 0.:    
                     lons_mw.append(cml.metadata['lon_center'])
                     lats_mw.append(cml.metadata['lat_center'])
                     values_mw.append(precip)                                                                    
                    
        if not all(v==0.0 for v in values_mw):                         
            if int_type == 'IDW':       
                interpol=mapping.inv_dist(lons_mw,lats_mw,values_mw,
                                              gridx,gridy,power,smoothing)
                                  
            elif int_type == 'Kriging':
                interpol=mapping.kriging(lons_mw,lats_mw,values_mw,gridx,gridy, 
                                         krig_type,variogram_model,drift_terms)                                      
            else:
                ValueError('Interpolation method not supported')                             
                                                                            
        if time is None:    
            if not all(v==0.0 for v in values_mw):                                                    
                cs = plt.contourf(gridx,gridy,interpol,levels=levels_sum,cmap=plt.cm.winter_r,transform=cartopy.crs.PlateCarree())
                cbar = plt.colorbar(cs,orientation='vertical', shrink=0.4)
                cbar.set_label('mm')
            plt.title('accumulated rainfall from time period: '+(self.set[0].data.index[0]).strftime('%Y-%m-%d %H:%M')+'UTC - '+
                        (self.set[0].data.index[-1]).strftime('%Y-%m-%d %H:%M')+'UTC',loc='right')

        else:
            if not all(v==0.0 for v in values_mw):
                cs = plt.contourf(gridx,gridy,interpol,levels=levels_rr,cmap=plt.cm.winter_r,alpha=0.6,transform=cartopy.crs.PlateCarree())
                cbar = plt.colorbar(cs,orientation='vertical', shrink=0.4)
                cbar.set_label('mm/h')            
            plt.title((pd.Timestamp(time)).strftime('%Y-%m-%d %H:%M')+'UTC',loc='right')

       
        if out_file is not None:
            plt.savefig(out_file,bbox_inches='tight',pad_inches=0)
=======
                               plist.append(cml.data['R_' + pair_id].resample('H',how='mean').cumsum()[-1])
                           if method == 'mean':
                               precip = np.mean(plist)                  
                           elif method == 'max':    
                               precip = np.max(plist)                     
                           elif method == 'min':
                               precip = np.min(plist)    
                           elif method in cml.processing_info['tx_rx_pairs']:
                               if 'R_' + pair_id in cml.data.keys():
                                   precip = cml.data['R_' + pair_id].resample('H',how='mean').cumsum()[-1]
                               else:
                                   print 'Warning: Pair ID '+method+' not available for link '+\
                                     cml.metadata['link_id'] + ' (link is ignored)'
                                   precip = None                                 
                           else: 
                               print 'Error: '+method+' not available for link '+\
                                         cml.metadata['link_id'] + ' (link is ignored)'
                               print '      Use "mean","max","min" or pair_id'            
                               precip = None
                                 
                           lons_mw.append(cml.metadata['lon_center'])
                           lats_mw.append(cml.metadata['lat_center'])
                           if precip >= 0.:
                               values_mw.append(precip)     
                           else:
                               values_mw.append(np.nan)
                               
                       else:
                           start = pd.Timestamp(time) - pd.Timedelta('10s')
                           stop = pd.Timestamp(time) + pd.Timedelta('10s')
                           plist = []
                           if start > cml.data.index[0] and stop < cml.data.index[-1]:
                                   for pair_id in cml.processing_info['tx_rx_pairs']:
                                       plist.append((cml.data['R_'+pair_id][start:stop]).values[0])   
                                     
                                   if method == 'mean':
                                       precip = np.mean(plist)                     
                                   elif method == 'max':
                                       precip = np.max(plist)                        
                                   elif method == 'min':
                                       precip = np.min(plist)                          
                                   elif method in cml.processing_info['tx_rx_pairs']:
                                       if 'R_' + pair_id in cml.data.keys():
                                           precip = (cml.data['R_'+method][start:stop]).values[0]
                                       else:
                                           print 'Warning: Pair ID '+method+' not available for link '+\
                                                     cml.metadata['link_id'] + ' (link is ignored)'
                                           precip = None
                                   else:
                                       print 'Error: '+method+' not available for link '+\
                                                 cml.metadata['link_id'] + ' (link is ignored)'
                                       print '      Use "mean","max","min" or pair_id'        
                                       precip = None
                                   
                                  
                                   lons_mw.append(lon_center)
                                   lats_mw.append(lat_center)
                                   if precip >= 0.:
                                       values_mw.append(precip)  
                                   else:
                                       values_mw.append(np.nan)    

                                
                           else:
                                   print 'Warning: Selected time not in data for link '+cml.metadata['link_id'] + \
                                                ' (link is ignored)'
                                   print '        Selectable time span:',cml.data.index[0], \
                                                                             cml.data.index[-1] 
                                   precip=None                               
                      
         
        val_mw=np.ma.compressed(np.ma.masked_where(np.isnan(values_mw),values_mw))
        lon_mw=np.ma.compressed(np.ma.masked_where(np.isnan(values_mw),lons_mw))
        lat_mw=np.ma.compressed(np.ma.masked_where(np.isnan(values_mw),lats_mw))
        

                                       
        if int_type == 'IDW':   
            interpol=mapping.inv_dist(lon_mw,lat_mw,val_mw,
                                      gridx,gridy,power,smoothing,nn)
                                  
        elif int_type == 'Kriging':
            interpol=mapping.kriging(lon_mw,lat_mw,val_mw,gridx,gridy, 
                                     krig_type,variogram_model,drift_terms)                                      
        else:
            ValueError('Interpolation method not supported')                             
        
        self.set_info['interpol'] = interpol
        self.set_info['interpol_grid'] = grid

        
        if plot:     
            fig = plt.figure(figsize=figsize)
            ax = plt.axes(projection=cartopy.crs.PlateCarree())
            gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')        
            gl.xlabels_top = False
            ax.set_extent((self.set_info['area'][0]-.05, self.set_info['area'][1]+.05,
                           self.set_info['area'][2]-.05, self.set_info['area'][3]+.05),
                             crs=cartopy.crs.PlateCarree())
            gg_tiles = img_tiles.MapQuestOSM()
    
            ax.add_image(gg_tiles, 11)          
            
             
            nws_precip_colors = [
            "#04e9e7",  # 0.01 - 0.10 inches
            "#019ff4",  # 0.10 - 0.25 inches
            "#0300f4",  # 0.25 - 0.50 inches
            "#02fd02",  # 0.50 - 0.75 inches
            "#01c501",  # 0.75 - 1.00 inches
            "#008e00",  # 1.00 - 1.50 inches
            "#fdf802",  # 1.50 - 2.00 inches
            "#e5bc00",  # 2.00 - 2.50 inches
            "#fd9500",  # 2.50 - 3.00 inches
            "#fd0000",  # 3.00 - 4.00 inches
            "#d40000",  # 4.00 - 5.00 inches
            "#bc0000",  # 5.00 - 6.00 inches
            "#f800fd",  # 6.00 - 8.00 inches
            "#9854c6"  # 10.00+
            ]
    
            precip_colormap = matplotlib.colors.ListedColormap(nws_precip_colors) 
            levels_rr = [0.01,0.1,0.25,0.5, 1.0, 1.5,2.0, 2.5,3.0, 5.0, 7.5, 10.0,12.5,15.0,20.0]
            levels_sum = [5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,75.0,100.0,150.0,200.0,250.0] 
            
            for cml in self.set:
                if 'site_A' in cml.metadata and 'site_B' in cml.metadata:
                    if 'lat' in cml.metadata['site_A'] and \
                       'lon' in cml.metadata['site_A'] and \
                       'lat' in cml.metadata['site_B'] and \
                       'lon' in cml.metadata['site_B']:
                           plt.plot([cml.metadata['site_A']['lon'],cml.metadata['site_B']['lon']],
                             [cml.metadata['site_A']['lat'],cml.metadata['site_B']['lat']],
                             linewidth=1,color='k',
                             transform=cartopy.crs.Geodetic()) 
                                                   
            if time is None:  
                interpolm = np.ma.masked_less(interpol, 1.)
                norm = matplotlib.colors.BoundaryNorm(levels_sum, 14)
                cs = plt.pcolormesh(gridx,gridy,interpolm, norm=norm, cmap=precip_colormap,alpha=0.4)                                                     
                #cs = plt.contourf(gridx,gridy,interpol,levels=levels_sum,
                #                  cmap=plt.cm.winter_r,transform=cartopy.crs.PlateCarree())
                cbar = plt.colorbar(cs,orientation='vertical', shrink=0.4)
                cbar.set_label('mm')
                plt.title('accumulated rainfall from time period: '+(self.set[0].data.index[0]).strftime('%Y-%m-%d %H:%M')+'UTC - '+
                            (self.set[0].data.index[-1]).strftime('%Y-%m-%d %H:%M')+'UTC',loc='right')
    
            else:
               interpolm = np.ma.masked_less(interpol,0.01)
               norm = matplotlib.colors.BoundaryNorm(levels_rr, 14)
               cs = plt.pcolormesh(grid[0],grid[1],interpolm, norm=norm, cmap=precip_colormap,alpha=0.4)
               #cs = plt.contourf(gridx,gridy,interpol,levels=levels_rr,
                               # cmap=plt.cm.winter_r,alpha=0.6,transform=cartopy.crs.PlateCarree())                
               cbar = plt.colorbar(cs,orientation='vertical', shrink=0.4)
               cbar.set_label('mm/h')            
               plt.title((pd.Timestamp(time)).strftime('%Y-%m-%d %H:%M')+'UTC',loc='right')
    
    
            plt.plot([self.set_info['area'][0],self.set_info['area'][0]],
                     [self.set_info['area'][2],self.set_info['area'][3]],linewidth=2,
                        color='k',alpha=0.6, transform=cartopy.crs.Geodetic())
            plt.plot([self.set_info['area'][0],self.set_info['area'][1]],
                     [self.set_info['area'][2],self.set_info['area'][2]],linewidth=2,
                        color='k',alpha=0.6, transform=cartopy.crs.Geodetic())
            plt.plot([self.set_info['area'][1],self.set_info['area'][1]],
                     [self.set_info['area'][2],self.set_info['area'][3]],linewidth=2,
                        color='k',alpha=0.6, transform=cartopy.crs.Geodetic())                    
            plt.plot([self.set_info['area'][0],self.set_info['area'][1]],
                     [self.set_info['area'][3],self.set_info['area'][3]],linewidth=2,
                        color='k',alpha=0.6, transform=cartopy.crs.Geodetic())
                        
            if out_file is not None:
                plt.savefig(out_file,bbox_inches='tight',format='png')
>>>>>>> 6356332... Revision of plotting and spatial interpolation methods. IDW now with limitation to nn neighbors. Plotting in spat_interpol now optional

                 