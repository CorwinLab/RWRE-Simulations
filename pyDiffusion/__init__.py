""" 
This scrip generates the namespace for libDiffusion
"""

try:
    import npquad
except ImportError as ie:
    raise type(ie)(
        str(ie) + ", npquad is available at "
        "http://www.github.com/SimonsGlass/numpy_quad"
        )

__all__ = [ "DiffusionPDF", "DiffusionTimeCDF", "DiffusionPositionPDF", "fileIO", "quadMath"]

from .pydiffusionCDF import DiffusionPositionCDF, DiffusionTimeCDF
from .pydiffusionPDF import DiffusionPDF
from . import fileIO
from . import quadMath