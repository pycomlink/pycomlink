import h5py
import xarray as xr
from tqdm import tqdm
from pycomlink.spatial.helper import haversine


def read_cmlh5_file_to_xarray(filename):
    """read a cmlh5 file and parse data from each cml_id to a xarray dataset

    Parameters
    ----------
    filename : string
        filename of a cmlh5 file

    Returns
    -------
    list
        list of xarray datasets

    """

    fh = h5py.File(name=filename, mode="r")

    id_list = list(fh.keys())

    cml_id = id_list[0]

    cml_g = fh[cml_id]

    ds_list = []
    for cml_id in tqdm(id_list):
        cml_g = fh[cml_id]
        ds_channel_list = []
        for channel_name, channel_g in cml_g.items():
            cml_ch_g = cml_g[channel_name]
            ds_temp = xr.Dataset(
                data_vars={
                    "tsl": ("time", cml_ch_g["tx"][:]),
                    "rsl": ("time", cml_ch_g["rx"][:]),
                },
                coords={
                    "time": (cml_ch_g["time"][:] * 1e9).astype("datetime64[ns]"),
                    "channel_id": channel_name,
                    "cml_id": cml_g.attrs["cml_id"],
                    "site_a_latitude": cml_g.attrs["site_a_latitude"],
                    "site_b_latitude": cml_g.attrs["site_b_latitude"],
                    "site_a_longitude": cml_g.attrs["site_a_longitude"],
                    "site_b_longitude": cml_g.attrs["site_b_longitude"],
                    "frequency": cml_ch_g.attrs["frequency"] / 1e9,
                    "polarization": cml_ch_g.attrs["polarization"],
                    "length": haversine(
                        cml_g.attrs["site_a_latitude"],
                        cml_g.attrs["site_a_longitude"],
                        cml_g.attrs["site_b_latitude"],
                        cml_g.attrs["site_b_longitude"],
                    ),
                },
            )
            ds_channel_list.append(ds_temp)
        ds_list.append(xr.concat(ds_channel_list, dim="channel_id"))
    return ds_list
