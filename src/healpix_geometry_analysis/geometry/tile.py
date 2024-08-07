import dataclasses
from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from healpix_geometry_analysis.coordinates import HealpixCoordinates

DIRECTIONS = ["p", "m"]
DIRECTION_T = Literal[*DIRECTIONS]

DISTANCE = ["chord_squared", "minus_cos_arc"]
DISTANCE_T = Literal[*DISTANCE]


@dataclasses.dataclass(kw_only=True)
class TileGeometry:
    """Distance problem for two opposite edges of a Healpix tile

    Parameters
    ----------
    coord : HealpixCoordinates
        Healpix coordinates object
    k_center : float
        NW-SE diagonal index of the pixel center
    kp_center : float
        NE-SW diagonal index of the pixel center
    direction : {"p", "m"}
        direction of edges of the tile to compare:
        - "p" (plus) for NE and SW edges
        - "m" (minus) for NW and SE edges
    distance : {"chord_squared", "minus_cos_arc"}
        Distance function to use:
        - "chord_squared" for squared chord distance in the unit sphere
        - "minus_cos_arc" for minus cosine of the great circle arc distance
    """

    delta: float = 0.5
    """Offset in the diagonal index from the center to the pixel, typically 0.5"""

    coord: HealpixCoordinates
    """Healpix coord object specifying order"""

    k_center: float
    """NW-SE diagonal indexx"""

    kp_center: float
    """NE-SW diagonal index"""

    direction: DIRECTION_T
    """direction of edges of the tile to compare, "p" (plus) or "m" (minus)"""

    distance: DISTANCE_T
    """Distance function to use, "chord_squared" or "minus_cos_arc\""""

    def __post_init__(self):
        assert (
            self.direction in DIRECTIONS
        ), f"Invalid direction: {self.direction}, must be one of {DIRECTIONS}"

    @classmethod
    def from_order(
        cls, order: int, *, k_center: float, kp_center: float, direction: DIRECTION_T, distance: DISTANCE_T
    ) -> Self:
        """Create TileProblem using order and diagonal indices

        Parameters
        ----------
        order : int
            Healpix order (depth) of the coord
        k_center : float
            NW-SE diagonal index of the pixel center
        kp_center : float
            NE-SW diagonal index of the pixel center
        direction : {"p", "m"}
            direction of edges of the tile to compare:
            - "p" (plus) for NE and SW edges
            - "m" (minus) for NW and SE edges
        distance : {"chord_squared", "minus_cos_arc"}
            Distance function to use:
            - "chord_squared" for squared chord distance in the unit sphere
            - "minus_cos_arc" for cosine of the great circle arc distance

        Returns
        -------
        TileProblem
            TileProblem object
        """
        return cls(
            coord=HealpixCoordinates.from_order(order),
            k_center=k_center,
            kp_center=kp_center,
            direction=direction,
            distance=distance,
        )

    parameter_names = ["k1", "k2", "kp1", "kp2"]

    @property
    def frozen_parameters(self) -> dict[str, object]:
        """Frozen parameters for the problem

        Returns
        -------
        tuple[str, str]
            Freezed parameters.
            ("k1" and "k2") for "m" direction
            and ("kp1" and "kp2") for "p" direction
        """
        if self.direction == "p":
            return {"kp1": self.kp_center - self.delta, "kp2": self.kp_center + self.delta}
        if self.direction == "m":
            return {"k1": self.k_center - self.delta, "k2": self.k_center + self.delta}
        raise ValueError(f"Invalid direction: {self.direction}, must be one of {DIRECTIONS}")

    @property
    def free_parameter_limits(self) -> dict[str, tuple[object, object]]:
        """Free parameters for the problem and their limits

        Returns
        -------
        dict[str, tuple[object, object]]
            Free parameters and their lower and upper limits.
            ("k1" and "k2") for "m" direction
            and ("kp1" and "kp2") for "p" direction
        """
        if self.direction == "p":
            limits = (self.k_center - jnp.abs(self.delta), self.k_center + jnp.abs(self.delta))
            return dict.fromkeys(["k1", "k2"], limits)
        if self.direction == "m":
            limits = (self.kp_center - jnp.abs(self.delta), self.kp_center + jnp.abs(self.delta))
            return dict.fromkeys(["kp1", "kp2"], limits)
        raise ValueError(f"Invalid direction: {self.direction}, must be one of {DIRECTIONS}")

    @property
    def free_parameter_distributions(self) -> dict[str, dist.Distribution]:
        """Free parameters for the problem and their distributions

        Returns
        -------
        dict[str, dist.Distribution]
            Free parameters and their distributions.
            ("k1" and "k2") for "m" direction
            and ("kp1" and "kp2") for "p" direction
        """
        return {name: dist.Uniform(*limits) for name, limits in self.free_parameter_limits.items()}

    def initial_params(self, rng_key: jax.random.PRNGKey) -> dict[str, object]:
        """Initial parameter values

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            Random number generator key

        Returns
        -------
        dict[str, object]
            Initial parameter values, free parameters are sampled from
            the uniform distribution within their limits, and frozen
            parameters are set to their values.
        """
        random_free_params = {}
        for name, distribution in self.free_parameter_distributions.items():
            random_free_params[name] = distribution.sample(rng_key)
            rng_key = jax.random.split(rng_key)[0]
        all_params = random_free_params | self.frozen_parameters
        return {name: all_params[name] for name in self.parameter_names}

    @property
    def limits(self) -> dict[str, tuple[object, object]]:
        """Limits for the parameters

        Returns
        -------
        dict[str, tuple[lower, upper]]
            Limits for the parameters. Frozen parameters have infinite limits.
        """
        frozen_parameters = dict.fromkeys(self.frozen_parameters, (-jnp.inf, jnp.inf))
        all_parameters = frozen_parameters | self.free_parameter_limits
        return {name: all_parameters[name] for name in self.parameter_names}

    @property
    def lower_bounds(self) -> dict[str, object]:
        """Lower limits for the parameters

        Returns
        -------
        dict[str, object]
            Lower limits for the parameters. Frozen parameters have -inf limits.
        """
        return {name: limits[0] for name, limits in self.limits.items()}

    @property
    def upper_bounds(self) -> dict[str, object]:
        """Upper limits for the parameters

        Returns
        -------
        dict[str, object]
            Upper limits for the parameters. Frozen parameters have inf limits.
        """
        return {name: limits[1] for name, limits in self.limits.items()}

    def calc_distance(self, k1, k2, kp1, kp2):
        """Calculate distance between two points

        The distance measure is defined by the distance attribute.
        It always grows with the Euclidean distance between the points.

        Parameters
        ----------
        k1 : float
            NW-SE diagonal index of the first pixel
        k2 : float
            NW-SE diagonal index of the second pixel
        kp1 : float
            NE-SW diagonal index of the first pixel
        kp2 : float
            NE-SW diagonal index of the second pixel

        Returns
        -------
        float
            Distance between the two pixels
        """
        if self.distance == "chord_squared":
            return self.coord.chord_squared(k1, kp1, k2, kp2)
        if self.distance == "minus_cos_arc":
            return -self.coord.cos_arc(k1, kp1, k2, kp2)
        raise ValueError(f"Invalid distance: {self.distance}, must be one of {DISTANCE}")

    def arc_length_radians(self, value):
        """Transform distance value returned by the model to radians"""
        if self.distance == "chord_squared":
            return 2.0 * jnp.arcsin(0.5 * jnp.sqrt(value))
        if self.distance == "minus_cos_arc":
            return jnp.arccos(value)
        raise ValueError(f"Invalid distance: {self.distance}, must be one of {DISTANCE}")

    def arc_length_degrees(self, value):
        """Transform distance value returned by the model to degrees"""
        return jnp.degrees(self.arc_length_radians(value))
