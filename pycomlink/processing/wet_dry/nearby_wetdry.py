import xarray as xr
import numpy as np
from tqdm import tqdm
import pycomlink.spatial.helper as spatial


def calc_distance_between_cml_endpoints(
    cml_ids,
    site_a_latitude,
    site_a_longitude,
    site_b_latitude,
    site_b_longitude,
):
    """
    calculating the distance between all start and endpoints of a network
    of CMLs using the Haversine distance forumla
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
    return ds


def instanteanous_to_minmax_data(ds_cml, interval=15, timeperiod=24, min_hours=6):
    """
    calculating pmin from instanteanousy measured rsl and tsl values
    ----------
    ds_cml : xarray.Dataset
         Time series of rsl and tsl
    interval : int
        Interval of pmin in minutes
    timeperiod : int
        Number of previous hours over which max(Pmin) is to be computed
    min_hours : int
        Minimum number of hours needed to compute max(Pmin)
    Returns
    -------
    xarray.Dataset
        Time series of pmin, max_pmin, deltaP and deltaPL
    References
    ----------
    .. [1] Overeem, A., Leijnse, H., and Uijlenhoet, R.: Retrieval algorithm
    for rainfall mapping from microwave links in a cellular communication network,
    Atmos. Meas. Tech., 9, 2425–2444, https://doi.org/10.5194/amt-9-2425-2016, 2016.
    """
    ds_cml["pmin"] = ds_cml.rsl - ds_cml.tsl
    ds_cml_minmax = ds_cml.pmin.resample(time=str(interval) + "min").min().to_dataset()

    # rolling window * 60min/interval(in minutes)
    period = int(timeperiod * 60 / interval)
    # min hours for calculation * 60min/interval(in minutes)
    hours_needed = int(min_hours * 60 / interval)
    ds_cml_minmax["max_pmin"] = ds_cml_minmax.pmin.rolling(
        time=period,
        min_periods=hours_needed,
    ).max(skipna=False)

    ds_cml_minmax["deltaP"] = ds_cml_minmax.pmin - ds_cml_minmax.max_pmin
    ds_cml_minmax["deltaPL"] = ds_cml_minmax["deltaP"] / ds_cml_minmax.length

    return ds_cml_minmax


def nearby_wetdry(
        ds_cml,
        ds_dist,
        r=15,
        thresh_median_P=-2.0,
        thresh_median_PL=-0.3,
        min_links=3,

):
    """
    calculating pmin from instanteanousy measured rsl and tsl values
    ----------
    ds_selected_cml : xarray.Dataset
         Time series of minmax values the CML selected for wet-dry calssification.
    ds_selected_cml : xarray.Dataset
         Time series of minmax values from all CMLs.
    ds_dist : xarray.Dataset
         Distance matrix between all CML endpoints.
    r : float
        Both end points are within a chosen radius r from either end of
        the already selected link are selected as well.
    thresh_median_P : float
        Threshold for median_P. Is dependent on the spatial correlation of rainfall.
    thresh_median_PL : float
        Threshold for median_PL. Is dependent on the spatial correlation of rainfall.
    min_links : int
        minimum number of CMLs within r needed to perform wet-dry classification
    Returns
    -------
    xarray.Dataset
        Time series wet-dry classification
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

    wet = xr.full_like(ds_cml.pmin, np.nan)

    for cmlid in tqdm(ds_cml.cml_id):
        # only make wet dry detection if min_links is reachied within r
        if sum(ds_dist.within_r.sel(cml_id1=cmlid).values) > min_links:
            # select all CMLs within r
            ds_nearby_cmls = ds_cml.isel(
                cml_id=ds_dist.within_r.sel(cml_id1=cmlid).values
            )

            # add selected cml if longer than r
            if ds_cml.sel(cml_id=cmlid).length > r:
                ds_nearby_cmls = xr.concat(
                    [ds_nearby_cmls, ds_cml.sel(cml_id=cmlid)],
                    dim="cml_id"
                )

            # calculate median delta P and delta PL for each timestep
            # .where checks if the minimal require number of CMLs has data
            medianP = (
                ds_nearby_cmls.deltaP.where(
                    ds_nearby_cmls.deltaP.count("cml_id") > min_links)
                    .median(dim="cml_id", skipna=False)
                    .values
            )
            medianPL = (
                ds_nearby_cmls.where(
                    ds_nearby_cmls.deltaP.count("cml_id") > min_links)
                    .deltaPL.median(dim="cml_id", skipna=False)
                    .values
            )

            ## actual wet dry classification
            wet.loc[dict(cml_id=cmlid)] = xr.where(
                cond=(np.isnan(medianP)) | (np.isnan(medianPL)),
                x=np.nan,
                y=((medianP < thresh_median_P) & (medianPL < thresh_median_PL)),
            )

            # if maxpim - pmin > 2db set two timesteps before and one aftere to wet
            diff2db_rule = (
                    ds_nearby_cmls.sel(cml_id=cmlid).max_pmin
                    - ds_nearby_cmls.sel(cml_id=cmlid).pmin
                    > 2
            )
            for shift in [-2, -1, 1]:
                wet.loc[dict(cml_id=cmlid)] = xr.where(
                    diff2db_rule.shift(time=shift) > 2,
                    x=1,
                    y=wet.loc[dict(cml_id=cmlid)],
                )
    return wet
