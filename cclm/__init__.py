try:
    from importlib.metadata import version
except ImportError:
    # compatibility for python <3.8
    from importlib_metadata import version

from importlib.metadata import version

__version__ = version(__package__)