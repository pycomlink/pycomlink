# pycomlink/processing/tensorflow_utils.py

import subprocess
import os
import sys

tf = None

def detect_gpu_without_tf():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        print("[✓] GPU detected via nvidia-smi.")
        return True
    except Exception:
        print("[ℹ️] No GPU detected.")
        return False

def get_tf(auto_install=False):
    global tf
    if tf is None:
        tf = ensure_tensorflow_installed(auto_install=auto_install)
        print_tf_device_info(tf)
    return tf

def ensure_tensorflow_installed(auto_install=False):
    try:
        import tensorflow as tf
        return tf
    except ImportError as e:
        has_gpu = detect_gpu_without_tf()
        pkg = "tensorflow[and-cuda]" if has_gpu else "tensorflow"
        if auto_install or os.getenv("ALLOW_TF_AUTOINSTALL") == "1":
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            import tensorflow as tf
            return tf
        else:
            raise ImportError(f"TensorFlow not installed. Run: pip install {pkg}") from e

def print_tf_device_info(tf):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[✅] TensorFlow is using GPU: {len(gpus)} device(s).")
    else:
        print("[⚠️] TensorFlow is using CPU.")
