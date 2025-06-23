from joblib import Memory
from sklearn.datasets import fetch_openml

memory = Memory(location="cifar_cache_dir", verbose=0)


@memory.cache
def load_cifar_cached(version=1, as_frame=False):
    """
    Download (once) and cache CIFAR-10 from OpenML.
    Subsequent calls with the same arguments return the cached result.
    """
    return fetch_openml("CIFAR_10", version=version, as_frame=as_frame)
