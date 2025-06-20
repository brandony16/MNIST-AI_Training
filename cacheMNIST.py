# 1. import and configure a cache directory
from joblib import Memory
from sklearn.datasets import fetch_openml

memory = Memory(location='cache_dir', verbose=0)

@memory.cache
def load_mnist_cached(version=1, as_frame=False):
    """
    Download (once) and cache MNIST from OpenML.
    Subsequent calls with the same arguments return the cached result.
    """
    return fetch_openml('mnist_784', version=version, as_frame=as_frame)
