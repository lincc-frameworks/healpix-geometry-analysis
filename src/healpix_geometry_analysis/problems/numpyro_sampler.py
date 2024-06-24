import dataclasses
from typing import Literal

import numpyro

from healpix_geometry_analysis.problems.base import BaseProblem


@dataclasses.dataclass(kw_only=True)
class NumpyroSamplerProblem(BaseProblem):
    """Description of the problem for numpyro sampler

    Parameters
    ----------
    geometry : TileGeometry
        Tile geometry object
    track_arc_length : bool, optional
        Track arc distance in degrees, by default False
    """

    track_arc_length: bool = False
    """Track arc distance in degrees"""

    def side1(self):
        """Get k & k' numpyro samples for the first side of the tile

        It is NE for "p" direction and NW for "m" direction
        """
        return self._side(1)

    def side2(self):
        """Get k & k' numpyro samples for the second side of the tile

        It is SW for "p" direction and SE for "m" direction
        """
        return self._side(2)

    def _side(self, index: Literal[1, 2]) -> tuple[object, object]:
        k_name = f"k{index}"
        kp_name = f"kp{index}"

        if k_name in self.geometry.free_parameter_distributions:
            k = numpyro.sample(k_name, self.geometry.free_parameter_distributions[k_name])
        else:
            k = self.geometry.frozen_parameters[k_name]

        if kp_name in self.geometry.free_parameter_distributions:
            kp = numpyro.sample(kp_name, self.geometry.free_parameter_distributions[kp_name])
        else:
            kp = self.geometry.frozen_parameters[kp_name]
        return k, kp

    def model(self):
        """Numpyro model to maximize

        It would maximize the distance between the two sides of the tile.
        Samples are k1 & k2 or kp1 & kp2 depending on the direction.
        Numpyro's factor is -distance to minimize it.
        Numpyro's deterministic is used to track distance:
        - "distance" is for distance measure in use (see .geometry.distance)
        - "arc_length_degree" is for arc distance in degrees,
          if track_arc_length is True
        """
        k1, kp1 = self.side1()
        k2, kp2 = self.side2()

        distance = self.geometry.calc_distance(k1, k2, kp1, kp2)
        # Use negative distance to minimize it
        numpyro.factor("target", -distance)

        # Track distance for diagnostics
        numpyro.deterministic("distance", distance)

        if self.track_arc_length:
            arc_length_degree = self.geometry.arc_length_degrees(distance)
            numpyro.deterministic("arc_length_degree", arc_length_degree)
