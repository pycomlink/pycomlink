import importlib

FOUND_TORCH = importlib.util.find_spec("torch") is not None
FOUND_TORCHINFO = importlib.util.find_spec("torchinfo") is not None



if FOUND_TORCH:
    from . import run_inference
    from . import inference_utils

else:
    # If torch and torchinfo is not installed,
    # we raise an exception when trying to use these functions.
    raise Exception("PyTorch must be installed to use 'pytorch_util'. "
                        "The 'torch' package is missing.")