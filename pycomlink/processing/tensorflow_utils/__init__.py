# __init__.py inside pycomlink/processing/tensorflow_utils
import importlib

FOUND_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None

if FOUND_TENSORFLOW:
    from . import inference_utils, run_inference
    from .lazy_tf import get_tf  # expose get_tf at package level
else:
    print(
        "[ℹ️] TensorFlow not installed. TensorFlow-based functions will not be available."
    )
    print("[💡] To use TensorFlow features with optimal GPU compatibility:")
    print("    For conda/mamba users:")
    print("        conda install tensorflow=2.15.* cudnn=8.9.*")
    print("        # or")
    print("        mamba install tensorflow=2.15.* cudnn=8.9.*")
    print()
    print("    For pip users:")
    print("        pip install tensorflow==2.15.*")
    print("        # Note: cuDNN compatibility depends on your CUDA installation")
    print()
    print("    ⚠️  Recommended versions for stability:")
    print("        - TensorFlow 2.15.x or 2.16.x")
    print("        - cuDNN 8.9.x (avoid cuDNN 9.x for better compatibility)")
    print("        - CUDA 12.2 or 12.3")
