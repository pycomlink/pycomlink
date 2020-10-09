from builtins import str
import pandas as pd
import numpy as np
import pyproj


def write_to_wasim_input_file(
    cml_list,
    fn,
    channel_name="channel_1",
    source_projection=None,
    target_projection=None,
):
    """Write hourly CML rain rates to CSV file in WaSiM input file format

    Parameters
    ----------

    cml_list : list
        List of Comlink objects
    fn : str
        Filename
    channel_name : str, optional
        Name of ComlinkChannel in Comlink object, defaults to 'channel_1'
    source_projection : int, optional
        EPSG projection number of coordinates in CML metadata
    target_projection : int, optional
        EPSG projection number of coordinates in CSV file
    """

    # Build DataFrame of rain rates for each CML
    df = pd.DataFrame()

    for cml in cml_list:
        df[cml.metadata["cml_id"]] = (
            cml.channels[channel_name].data.R.resample("H", label="right").mean()
        )
    df["YYYY"] = df.index.year
    df["MM"] = np.char.mod("%02d", df.index.month)
    df["DD"] = np.char.mod("%02d", df.index.day)
    df["HH"] = np.char.mod("%02d", df.index.hour)
    df.loc[df["HH"] == "00", "HH"] = "24"

    # Reorder columns so that date columns come first
    cols_time = ["YYYY", "MM", "DD", "HH"]
    cols = cols_time + [
        col_name for col_name in df.columns if col_name not in cols_time
    ]
    df = df[cols]

    # Build DataFrame for coordinates and altitude (missing in current metadata)
    df_coords = pd.DataFrame()
    for date_str in ["YYYY", "MM", "DD", "HH"]:
        df_coords[date_str] = 3 * [date_str]
    for cml in cml_list:
        if (source_projection is None) and (target_projection is None):
            x, y = cml.get_center_lon_lat()
        elif (source_projection is not None) and (target_projection is not None):
            in_proj = pyproj.Proj(init="epsg:" + str(source_projection))
            out_proj = pyproj.Proj(init="epsg:" + str(target_projection))
            x, y = pyproj.transform(in_proj, out_proj, *cml.get_center_lon_lat())
        else:
            raise ValueError(
                "`source_projection` and `target_projection` "
                "must both be either None or a EPSG string"
            )
        altitude = -9999
        df_coords[cml.metadata["cml_id"]] = [altitude, x, y]

    # Write variable info to file
    with open(fn, "w") as f:
        f.write("Precipitation mm\n")
    # Write coordinates to file
    df_coords.to_csv(fn, mode="a", index=False, sep=" ", na_rep="-9999.0", header=False)
    # Write rain rates to file
    df.to_csv(
        fn, mode="a", index=False, sep=" ", na_rep="-9999.0", float_format="%2.1f"
    )
