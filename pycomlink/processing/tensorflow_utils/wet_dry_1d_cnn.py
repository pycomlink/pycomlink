import xarray as xr

def wet_dry_1d_cnn(
    ds: xr.Dataset,
    input_vars=["tl1", "tl2"],
    seq_len=180,
    prob_name="CNN",
    threshold=0.1,
    batch_size=128,
    lr=0.05,
    json_url="https://github.com/toufikshit/pycomlink/releases/download/v1/CNN__model_v0_cz.json",
    weights_url="https://github.com/toufikshit/pycomlink/releases/download/v1/CNN__model_v0_cz.weights.h5",
    cache_dir="model_cnn",
    force_download=False,
    return_ds=True
):
    """
    Run the 1D CNN wet/dry prediction.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing 'tl1' and 'tl2' (or other input_vars).
    input_vars : list of str
        Variables to use as model input.
    seq_len : int
        Sequence length used for CNN input.
    model_name : str
        Name used for output DataArray.
    threshold : float
        Threshold for binary prediction (default: 0.1).
    batch_size : int
        Batch size for model prediction.
    lr : float
        Learning rate for SGD optimizer.
    json_url : str
        URL to model architecture (.json).
    weights_url : str
        URL to weights (.h5).
    cache_dir : str
        Directory to cache model files.
    force_download : bool
        If True, re-download model files even if cached.
    return_ds : bool
        If True, return merged dataset. If False, return just the prediction DataArray.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Dataset with prediction added or just the prediction array.
    """
    import pycomlink.processing.tensorflow_utils.cnn as cnn

    tf = cnn.get_tf()

    # Load model
    json_path, weights_path = cnn.resolve_model_paths(
        json_source=json_url,
        weights_source=weights_url,
        cache_dir=cache_dir,
        force_download=force_download
    )
    model = cnn.load_model_from_local(json_path, weights_path, lr=lr)

    # Get input CML IDs
    cml_ids = ds.cml_id.values

    # Prepare samples and scale
    X, ids = cnn.create_samples(ds=ds, cml_ids=cml_ids, input_vars=input_vars, seq_len=seq_len)
    X_scaled, _ = cnn.scale_features(input_seq=X, input_vars=input_vars)

    # Predict
    y_prob, y_pred = cnn.run_inference(
        model=model,
        model_input=X_scaled,
        threshold=threshold,
        batch_size=batch_size
    )

    # Store predictions as DataArray
    pred_array = cnn.store_predictions(
        ds = ds,
        model_prob=y_prob,
        cml_ids=ids,
        var_name=prob_name)    

    if return_ds:
        return xr.merge([ds, pred_array.to_dataset()])
    else:
        return pred_array
