


from tqdm import tqdm
import xarray as xr
import numpy as np


def build_xr(cml_list):

    """ Generate a netCDF file from an cmlh5 file

     Parameters
    ----------
    cml_list: cml_list from cmlh5 file

    Returns
    ----------
    ds: xarray.Dataset
        xarray.Dataset containing the data from the cmlh5 data
    """


    # get first and last time stamp and generate DatetimeIndex #
    cml = cml_list[0]
    t_start = cml.channel_1.data.index.min()
    t_end = cml.channel_1.data.index.max()
    for cml in cml_list:
        t_cml_min = cml.channel_1.data.index.min()
        t_cml_max = cml.channel_1.data.index.max()
        if t_cml_min < t_start:
            t_start = t_cml_min
        if t_cml_max > t_end:
            t_end = t_cml_max
    t_index = pd.DatetimeIndex(freq='min',
                               start=t_start.floor('min'),
                               end=t_end.ceil('min'))

    ds_cml_dict = {}
    count_cml = 0
    for cml in tqdm(cml_list):
        cml_id = cml.metadata['cml_id']
        for ch_name, cml_ch in cml.channels.items():
            df_temp = cml_ch.data.copy()
            # drop rows with duplicated time stamp
            df_temp.index = df_temp.index.round('s')
            df_temp.index = df_temp.index.floor('min')
            s = pd.Series(df_temp.index)
            if len(s[s.duplicated()]) > 0:
                df_temp = df_temp.loc[~df_temp.index.duplicated(keep=False)]
            ds_temp = xr.Dataset.from_dataframe(df_temp.reindex(
                t_index).tz_localize(None))
            ds_temp = ds_temp.rename({'index': 'time'})
            ds_temp.coords['frequency'] = (['cml_id',
                                            'channel_id'],
                                           np.atleast_2d(
                                               [cml_ch.metadata['frequency'],
                                                ]))
            ds_temp.coords['polarization'] = (
                ['cml_id', 'channel_id'],
                np.atleast_2d([cml_ch.metadata['polarization'], ]))
            ds_temp.coords['cml_id'] = ('cml_id', [cml.metadata['cml_id'], ])
            ds_temp.coords['length'] = ('cml_id', [cml.metadata['length'], ])
            ds_temp.coords['site_a_latitude'] = ('cml_id',
                                                 [cml.metadata['site_a_latitude'], ])
            ds_temp.coords['site_a_longitude'] = ('cml_id',
                                                  [cml.metadata['site_a_longitude'], ])
            ds_temp.coords['site_b_latitude'] = ('cml_id',
                                                 [cml.metadata['site_b_latitude'], ])
            ds_temp.coords['site_b_longitude'] = ('cml_id',
                                                  [cml.metadata['site_b_longitude'], ])
            ds_temp.coords['channel_id'] = ('channel_id', [ch_name, ])

            try:
                ds_cml_dict[ch_name].append(ds_temp)
            except KeyError:
                ds_cml_dict[ch_name] = []
                ds_cml_dict[ch_name].append(ds_temp)

    print('concat temp_dataset')
    # Build a xarray.Dataset from the CML DataFrame lists of the two channels
    ds = xr.concat(
        objs=[xr.concat(ds_cml_dict['channel_1'], dim='cml_id'),
              xr.concat(ds_cml_dict['channel_2'], dim='cml_id')],
        dim='channel_id')

    return ds