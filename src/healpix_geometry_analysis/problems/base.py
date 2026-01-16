import dataclasses

from healpix_geometry_analysis.geometry.base import BaseGeometry


@dataclasses.dataclass
class BaseProblem:
    """Base class for the problems

    Parameters
    ----------
    geometry : BaseGeometry
        Geometry object
    """

    geometry: BaseGeometry
    """Geometry object describing k1, k2, kp1 & kp2 for the problem"""
