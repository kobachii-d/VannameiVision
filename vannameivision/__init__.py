import warnings

# Suppress the TensorFlow Addons warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow_addons")

# Now, your regular imports for this package
from .utils import build, read_preprocess, make_prediction, get_image_paths
