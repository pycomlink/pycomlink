import importlib

FOUND_TORCH = importlib.util.find_spec("torch") is not None


if FOUND_TORCH:
    from . import run_inference
    from . import inference_utils
    from . import pytorch_utils

else:
    # If torch and torchinfo is not installed,
    # we raise an exception when trying to use these functions.
    raise Exception(
        "pycomlink.pytorch_utility requires PyTorch.\n"
        "Install it with: `pip install pycomlink[torch]` or manualy using: `pip install torch`. "
    )
