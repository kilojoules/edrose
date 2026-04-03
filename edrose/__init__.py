"""Elliptical wind direction rose model.

Implementation of Hart (2025), "An elliptical parameterisation of the wind
direction rose", Wind Energy Science, 10, 1821-1827.
https://doi.org/10.5194/wes-10-1821-2025
"""

from edrose.model import EllipticalWindRose, MixtureEllipticalWindRose

__all__ = ["EllipticalWindRose", "MixtureEllipticalWindRose"]
__version__ = "0.1.0"
