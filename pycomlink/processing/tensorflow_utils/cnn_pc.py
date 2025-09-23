def pc_cnn_wet(merged, pc_var="pc", cnn_var="cnn_prob_TL"):
    """
    Adds wet/dry classification flags to a Dataset based on:
    - Precipitation thresholds on the `pc` variable.
    - CNN-derived probability threshold on the `cnn_prob` variable.

    This mimics the original script behavior exactly:
    - Adds: pc_wet01, pc_wet10, pc_wet20, pc_wet30 (based on pc_var)
    - Adds: cnn_wet82 (based on cnn_var)

    Parameters
    ----------
    merged : xr.Dataset
        Input dataset.
    pc_var : str
        Name of the precipitation variable (default: 'pc').
    cnn_var : str
        Name of the CNN probability variable (default: 'cnn_prob_TL').

    Returns
    -------
    xr.Dataset
        The same dataset with new binary variables added.
    """
    # Precipitation-based wet/dry classification
    merged['pc_wet01'] = merged[pc_var] >= 0.1
    merged['pc_wet10'] = merged[pc_var] >= 10
    merged['pc_wet20'] = merged[pc_var] >= 20
    merged['pc_wet30'] = merged[pc_var] >= 30

    # CNN-based wet/dry classification
    merged["cnn_wet82"] = merged[cnn_var] > 0.82

    return merged


def combine_wet_detections(cml, cnn_var="cnn_prob_TL", pc_var="pc", use_dask=False, chunk_dict=None):
    """
    Combines different wet/dry detection layers in the xarray Dataset `cml`
    using CNN wetness probability and PC Precipitation Probability product Class 3 (max,; 2024).

    Parameters
    ----------
    cml : xr.Dataset
        Input dataset.
    cnn_var : str
        Name of CNN probability variable (e.g., "cnn_prob_TL").
    pc_var : str
        Name of precipitation variable (e.g., "pc").
    use_dask : bool
        If True, dataset will be rechunked using Dask to handle large data.
    chunk_dict : dict or None
        Optional dictionary to define Dask chunks (e.g., {"time": 1000, "cml_id": 500}).

    Returns
    -------
    xr.Dataset
        Dataset with new combined detection layers.
    """
    import xarray as xr

    # ✅ Optional: enable Dask chunking
    if use_dask:
        if chunk_dict is None:
            chunk_dict = {"time": 1000, "cml_id": 500}  # default chunking
        cml = cml.chunk(chunk_dict)
        print(f"[⚙️] Dask enabled with chunks: {chunk_dict}")

    # ✅ Hardcoded wet variables based on prior convention
    pc_wet_vars = [f"{pc_var}_wet01", f"{pc_var}_wet10"]
    cnn_wet_var = "cnn_wet82"

    # ✅ First loop: combine pc_wet01 and pc_wet10
    for wet in pc_wet_vars:
        combined = wet + "_combined"
        cml[combined] = cml[wet]
        cml[combined] = xr.where(cml[f"{pc_var}_wet01"] == False, False, cml[combined])
        cml[combined] = xr.where(cml[cnn_var] > 0.94, True, cml[combined])
        cml[combined] = xr.where(cml[f"{pc_var}_wet30"] == True, True, cml[combined])
        cml[combined] = xr.where(cml[cnn_var] <= 0.1, False, cml[combined])

    # ✅ Second combination: cnn_wet82 + pc
    combined = f"{cnn_wet_var}_{pc_var}_combined"
    cml[combined] = cml[cnn_wet_var]
    cml[combined] = xr.where(cml[f"{pc_var}_wet01"] == False, False, cml[combined])
    cml[combined] = xr.where(cml[cnn_var] > 0.94, True, cml[combined])
    cml[combined] = xr.where(cml[f"{pc_var}_wet30"] == True, True, cml[combined])
    cml[combined] = xr.where(cml[cnn_var] <= 0.1, False, cml[combined])

    return cml
