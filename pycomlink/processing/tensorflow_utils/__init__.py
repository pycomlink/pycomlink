# __init__.py inside pycomlink/processing/tensorflow_utils
import importlib

FOUND_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None

if FOUND_TENSORFLOW:
    from .lazy_tf import get_tf          # expose get_tf at package level
    from . import inference_utils
    from . import run_inference
else:
    print("[ℹ️] TensorFlow not installed. TensorFlow-based functions will not be available.")
