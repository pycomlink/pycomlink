import xarray as xr
import numpy as np
from tqdm import tqdm
from ... spatial import helper as spatial


def calc_distance_between_cml_endpoints(
    cml_ids,
    site_a_latitude,
    site_a_longitude,
    site_b_latitude,
    site_b_longitude,
):
    """
    Calculating the distance from and to all start and endpoints of a network
    of CMLs using the Haversine distance formula. This includes the start and
    endpoint of each CML (which equals its length). The distance between start
    and endpoint of a CML will be set to 0 for the case when r (the radius for
    which CMLs are considered to be "nearby") is smaller than the actual length
    of the CML
    ----------
    cml_ids : list of str or int
         ids of CMLs
    site_a_latitude : list or array
        latitude values of site a
    site_a_longitude : list or array
        longitude values of site a
    site_b_latitude : list or array
        latitude values of site b
    site_b_longitude : list or array
        longitude values of site b
    Returns
    -------
    xarray.Dataset
        distances between all start and endpoints in four variables, a_to_all_a,
        a_to_all_b, b_to_all_a, b_to_all_b
    """

    n_cmls = len(cml_ids)
    ds = xr.Dataset(
        data_vars=dict(
            a_to_all_a=(["cml_id1", "cml_id2"], np.full([n_cmls, n_cmls], np.nan)),
            a_to_all_b=(["cml_id1", "cml_id2"], np.full([n_cmls, n_cmls], np.nan)),
            b_to_all_a=(["cml_id1", "cml_id2"], np.full([n_cmls, n_cmls], np.nan)),
            b_to_all_b=(["cml_id1", "cml_id2"], np.full([n_cmls, n_cmls], np.nan)),
        ),
        coords=dict(cml_id1=cml_ids, cml_id2=cml_ids),
    )
    for i, cmlid in tqdm(enumerate(cml_ids)):
        ds.loc[dict(cml_id1=cmlid)]["a_to_all_a"][:] = spatial.haversine(
            np.array(site_a_longitude[i]),
            np.array(site_a_latitude[i]),
            np.array(site_a_longitude),
            np.array(site_a_latitude),
        )
        ds.loc[dict(cml_id1=cmlid)]["a_to_all_b"][:] = spatial.haversine(
            np.array(site_a_longitude[i]),
            np.array(site_a_latitude[i]),
            np.array(site_b_longitude),
            np.array(site_b_latitude),
        )
        ds.loc[dict(cml_id1=cmlid)]["b_to_all_a"][:] = spatial.haversine(
            np.array(site_b_longitude[i]),
            np.array(site_b_latitude[i]),
            np.array(site_a_longitude),
            np.array(site_a_latitude),
        )
        ds.loc[dict(cml_id1=cmlid)]["b_to_all_b"][:] = spatial.haversine(
            np.array(site_b_longitude[i]),
            np.array(site_b_latitude[i]),
            np.array(site_b_longitude),
            np.array(site_b_latitude),
        )
    # set distance between start and endpoint of each CML to 0km
    # this has to be done so that CMLs are not rejected when the radius r
    # which defines nearby CMLs is larger than the distance between endpoints
    # of a CML (aka its length). Otherwise, the signal levels of CMLs with
    # length > r will not be considered in the nearby approach
    ds = xr.where(
        ds.cml_id1 == ds.cml_id2,
        0,
        ds)
    return ds


def instantaneous_to_minmax_data(rsl, tsl, length, interval=15, timeperiod=24, min_hours=6):
    """
    calculating pmin from instanteanousy measured rsl and tsl values
    ----------
    rsl : xarray.DataArray
         Time series of received signal levels (rsl)
    tsl : xarray.DataArray
         Time series of transmitted signal levels (tsl)
    length : xarray.Dataset,float64
        Lengths of CML(s) which are provided
    interval : int
        Interval of pmin in minutes
    timeperiod : int
        Number of previous hours over which max(pmin) is to be computed
    min_hours : int
        Minimum number of hours needed to compute max(pmin)
    Returns
    -------
    tuple of xarray.Datasets
        Time series of pmin, max_pmin, deltaP and deltaPL
    References
    ----------
    .. [1] Overeem, A., Leijnse, H., and Uijlenhoet, R.: Retrieval algorithm
    for rainfall mapping from microwave links in a cellular communication network,
    Atmos. Meas. Tech., 9, 2425–2444, https://doi.org/10.5194/amt-9-2425-2016, 2016.
    """
    pmin = rsl - tsl
    pmin = pmin.resample(time=str(interval) + "min").min()
    pmax = pmin.resample(time=str(interval) + "min").max()

    # rolling window * 60min/interval(in minutes)
    period = int(timeperiod * 60 / interval)
    # min hours for calculation * 60min/interval(in minutes)
    hours_needed = int(min_hours * 60 / interval)
    max_pmin = pmin.rolling(
        time=period,
        min_periods=hours_needed,
    ).max(skipna=False)

    deltaP = pmin - max_pmin
    deltaPL = deltaP / length

    return (
        pmin,
        max_pmin,
        deltaP,
        deltaPL,
    )


def nearby_wetdry(
        pmin,
        max_pmin,
        deltaP,
        deltaPL,
        ds_dist,
        r=15,
        thresh_median_P=-1.4,
        thresh_median_PL=-0.7,
        min_links=3,
        interval=15,
        timeperiod=24,
):
    """
    Classification of rainy and dry periods from diagnostic (min-max) CML signal
    levels following the nearby link approach.
    ----------
    pmin : xarray.DataArray
         Time series of pmin, must include cml_id and time as dimensions
    max_pmin: xarray.DataArray
        Time series of max_pmin, must include cml_id and time as dimensions
    deltaP : xarray.DataArray
        Time series of deltaP, must include cml_id and time as dimensions
    deltaPL : xarray.DataArray
        Time series of deltaPL, must include cml_id and time as dimensions
    ds_dist : xarray.Dataset
         Distance matrix between all CML endpoints calculated with
         `calc_distance_between_cml_endpoints()`, must include cml_id
    r : float
        Radius for which surrounding CMLs (both end points are within a chosen
        radius r from either end of the selected link)
    thresh_median_P : float
        Threshold for median_P. Is dependent on the spatial correlation of rainfall.
    thresh_median_PL : float
        Threshold for median_PL. Is dependent on the spatial correlation of rainfall.
    min_links : int
        minimum number of CMLs within r needed to perform wet-dry classification
    Returns
    -------
    tuple of four xarray.Datasets
        Time series of wet-dry classification as well as three variable for
        later quality control
    References
    ----------
    .. [1] Overeem, A., Leijnse, H., and Uijlenhoet, R.: Retrieval algorithm
    for rainfall mapping from microwave links in a cellular communication network,
    Atmos. Meas. Tech., 9, 2425–2444, https://doi.org/10.5194/amt-9-2425-2016, 2016.
    """
    # get number of CMLs within r for each CML
    ds_dist["within_r"] = (
            (ds_dist.a_to_all_a < r)
            & (ds_dist.a_to_all_b < r)
            & (ds_dist.b_to_all_a < r)
            & (ds_dist.b_to_all_b < r)
    )

    if not ((pmin.cml_id.values == max_pmin.cml_id.values).all() and (
            deltaP.cml_id.values == deltaPL.cml_id.values
    ).all() and (deltaP.cml_id.values == max_pmin.cml_id.values).all()):
        raise ValueError(
            "All input variables must contain the same cml_ids.")

    ds_cml = pmin.to_dataset(name='pmin')
    ds_cml["max_pmin"] = max_pmin
    ds_cml["deltaP"] = deltaP
    ds_cml["deltaPL"] = deltaPL

    wet = xr.full_like(ds_cml.pmin, np.nan)
    F = xr.full_like(ds_cml.pmin, np.nan)
    medianP_out = xr.full_like(ds_cml.pmin, np.nan)
    medianPL_out = xr.full_like(ds_cml.pmin, np.nan)

    for cmlid in tqdm(ds_cml.cml_id):
        # only make wet dry detection if min_links is reached within r
        if sum(ds_dist.within_r.sel(cml_id1=cmlid).values) > min_links:

            # select all CMLs within r
            ds_nearby_cmls = ds_cml.isel(
                cml_id=ds_dist.within_r.sel(cml_id1=cmlid).values
            )

            # calculate median delta P and delta PL for each timestep
            # .where checks if the minimal require number of CMLs has data
            medianP = (
                ds_nearby_cmls.deltaP.where(
                    ds_nearby_cmls.deltaP.count("cml_id") > min_links)
                .median(dim="cml_id", skipna=True)
                .values
            )
            medianP_out.loc[dict(cml_id=cmlid)] = medianP
            medianPL = (
                ds_nearby_cmls.where(
                    ds_nearby_cmls.deltaP.count("cml_id") > min_links)
                .deltaPL.median(dim="cml_id", skipna=True)
                .values
            )
            medianPL_out.loc[dict(cml_id=cmlid)] = medianPL

            # actual wet dry classification
            wet.loc[dict(cml_id=cmlid)] = xr.where(
                cond=(np.isnan(medianP)) | (np.isnan(medianPL)),
                x=np.nan,
                y=((medianP < thresh_median_P) & (medianPL < thresh_median_PL)),
            )

            # calculate the F values which can be used as outlier filter
            F_val = ds_nearby_cmls.deltaPL.sel(cml_id=cmlid) - medianPL
            F.loc[dict(cml_id=cmlid)] = F_val.rolling(time=timeperiod * 4).sum(
                skipna=True) * (interval / 60)

            wet_tmp = wet.copy()

            # if wet is true and deltaP < -2db then set two timesteps before and
            # one after to wet
            for shift in [1, -1, -2]:
                wet.loc[dict(cml_id=cmlid.values)] = xr.where(
                    ((wet_tmp.loc[dict(cml_id=cmlid)] == 1) & (
                            ds_nearby_cmls.sel(
                                cml_id=cmlid).deltaP < -2)).shift(
                        time=shift),  ##
                    # shift here
                    x=1,
                    y=wet.loc[dict(cml_id=cmlid.values)],
                )

            wet.loc[dict(cml_id=cmlid.values)] = xr.where(
                np.isnan(wet_tmp.loc[dict(cml_id=cmlid.values)]),
                np.nan,
                wet.loc[dict(cml_id=cmlid.values)]
            )
    return wet, F, medianP_out, medianPL_out