import dataclasses

import numpyro

from healpix_geometry_analysis.problems.base import BaseProblem


@dataclasses.dataclass(kw_only=True)
class NumpyroSamplerProblem(BaseProblem):
    """Description of the problem for numpyro sampler

    Parameters
    ----------
    geometry : BaseGeometry
        Tile geometry object
    track_arc_length : bool, optional
        Track arc distance in degrees, by default False
    """

    track_arc_length: bool = False
    """Track arc distance in degrees"""

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
        free_params = {
            name: numpyro.sample(name, dist)
            for name, dist in self.geometry.free_parameter_distributions.items()
        }
        params = free_params | self.geometry.frozen_parameters

        distance = self.geometry.calc_distance(params)
        # Use negative distance to minimize it
        numpyro.factor("target", -distance)

        # Track distance for diagnostics
        numpyro.deterministic("distance", distance)

        if self.track_arc_length:
            arc_length_degree = self.geometry.arc_length_degrees(distance)
            numpyro.deterministic("arc_length_degree", arc_length_degree)
