import xarray as xr
from pycomlink.processing.tensorflow_utils import inference_utils
from pycomlink.processing.tensorflow_utils import lazy_tf

DEFAULT_INPUT_VARS = ["tl1", "tl2"]
DEFAULT_SEQ_LEN = 180
DEFAULT_PROB_NAME = "CNN"
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 0.05
DEFAULT_JSON_URL = "https://github.com/toufikshit/pycomlink/releases/download/v1/CNN__model_v0_cz.json"
DEFAULT_WEIGHTS_URL = "https://github.com/toufikshit/pycomlink/releases/download/v1/CNN__model_v0_cz.weights.h5"
DEFAULT_CACHE_DIR = "model_cnn"


def wet_dry_1d_cnn(
    ds: xr.Dataset,
    id_coord: str = "cml_id",
    input_vars=None,
    seq_len: int = None,
    prob_name: str = None,
    batch_size: int = None,
    lr: float = None,
    json_url: str = None,
    weights_url: str = None,
    cache_dir: str = None,
    force_download: bool = False,
    return_ds: bool = True,
    cml_ids=None,
    install_tensorflow: bool = False
):
    """
    Run the 1D CNN wet/dry prediction.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing input variables and ID coordinate.
    id_coord : str, default="cml_id"
        Name of the coordinate in `ds` that holds the link/channel IDs.
    ...
    """

    tf = lazy_tf.get_tf(auto_install=install_tensorflow)

    # Set defaults if not provided
    if input_vars is None:
        input_vars = DEFAULT_INPUT_VARS
    if seq_len is None:
        seq_len = DEFAULT_SEQ_LEN
    if prob_name is None:
        prob_name = DEFAULT_PROB_NAME
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    if lr is None:
        lr = DEFAULT_LR
    if json_url is None:
        json_url = DEFAULT_JSON_URL
    if weights_url is None:
        weights_url = DEFAULT_WEIGHTS_URL
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    # Validate ID coordinate
    if (id_coord not in ds.coords and id_coord not in ds.variables and id_coord not in ds.dims):
        raise KeyError(f"'{id_coord}' not found in dataset. "
                   f"Please specify the correct coordinate name via `id_coord=`.")


    if cml_ids is None:
        cml_ids = ds[id_coord].values



    # Load model
    json_path, weights_path = inference_utils.resolve_model_paths(
        json_source=json_url,
        weights_source=weights_url,
        cache_dir=cache_dir,
        force_download=force_download
    )
    model = inference_utils.load_model_from_local(json_path, weights_path, lr=lr)

    # Prepare samples and scale
    X, ids, t_idx = inference_utils.create_samples(ds=ds, cml_ids=cml_ids, input_vars=input_vars, seq_len=seq_len)
    X_scaled = inference_utils.scale_features(input_seq=X, input_vars=input_vars)

    # Predict
    y_prob = inference_utils.run_inference(
        model=model,
        model_input=X_scaled,
        batch_size=batch_size
    )
    del X_scaled
    
    # Store predictions as DataArray
    pred_array = inference_utils.store_predictions(
        ds = ds,
        model_prob=y_prob,
        time_indices = t_idx,
        cml_ids=ids,
        var_name=prob_name)    

    if return_ds:
        return xr.merge([ds, pred_array.to_dataset()])
    else:
        return pred_array
