import warnings
from importlib.metadata import version, PackageNotFoundError
__pkgname__ = "mlforgex"

try:
    __version__ = version(__pkgname__)
except PackageNotFoundError:
    __version__ = "0.0.0"

try:
    import requests

    resp = requests.get(f"https://pypi.org/pypi/{__pkgname__}/json", timeout=2)
    if resp.status_code == 200:
        latest_version = resp.json()["info"]["version"]
        if __version__ < latest_version:
            warnings.warn(
                f"You are using {__pkgname__} {__version__}. "
                f"A newer version ({latest_version}) is available. "
                f"Please upgrade by running:\n"
                f"    pip install --upgrade {__pkgname__}",
                UserWarning
            )
except Exception:
    pass 

from .train import train_model
from .predict import predict
