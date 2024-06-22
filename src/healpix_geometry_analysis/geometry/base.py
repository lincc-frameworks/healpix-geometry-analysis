import dataclasses
from abc import ABC, abstractmethod
from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from healpix_geometry_analysis.coordinates import HealpixCoordinates

DIRECTIONS = ["p", "m"]
DIRECTION_T = Literal[*DIRECTIONS]

DISTANCE = ["chord_squared", "minus_cos_arc"]
DISTANCE_T = Literal[*DISTANCE]

REGION = ["equator", "polar"]
REGION_T = Literal[*REGION]


@dataclasses.dataclass(kw_only=True)
class BaseGeometry(ABC):
    """Distance problem for two opposite edges of a Healpix tile

    Parameters
    ----------
    coord : HealpixCoordinates
        Healpix coordinates object
    direction : {"p", "m"}
        direction of edges of the tile to compare:
        - "p" (plus) for NE and SW edges
        - "m" (minus) for NW and SE edges
    distance : {"chord_squared", "minus_cos_arc"}
        Distance function to use:
        - "chord_squared" for squared chord distance in the unit sphere
        - "minus_cos_arc" for minus cosine of the great circle arc distance
    """

    coord: HealpixCoordinates
    """Healpix coord object specifying order"""

    direction: DIRECTION_T
    """direction of edges of the tile to compare, "p" (plus) or "m" (minus)"""

    distance: DISTANCE_T
    """Distance function to use, "chord_squared" or "minus_cos_arc\""""

    def __post_init__(self):
        assert (
            self.direction in DIRECTIONS
        ), f"Invalid direction: {self.direction}, must be one of {DIRECTIONS}"

    @classmethod
    def from_order(cls, order: int, *, direction: DIRECTION_T, distance: DISTANCE_T) -> Self:
        """Create TileProblem using order and diagonal indices

        Parameters
        ----------
        order : int
            Healpix order (depth) of the coord
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
            direction=direction,
            distance=distance,
        )

    diagonal_names: tuple[str, str, str, str] = dataclasses.field(
        init=False, default=("k1", "k2", "kp1", "kp2")
    )

    @abstractmethod
    def diagonal_indices(self, params: dict[str, object]) -> tuple[object, object, object, object]:
        """Diagonal indices of the pixels

        Parameters
        ----------
        params : dict[str, object]
            Pytree with parameter values

        Returns
        -------
        tuple[object, object, object, object]
            Diagonal indices of the pixel: k1, k2, kp1, kp2
        """

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """Parameter names for the problem

        Returns
        -------
        tuple[str]
            Parameter names
        """
        raise NotImplementedError("To be implemented in subclasses")

    @property
    @abstractmethod
    def frozen_parameters(self) -> dict[str, object]:
        """Frozen parameters for the problem

        Returns
        -------
        dict[str, object]
            Freezed parameters and their values
        """
        raise NotImplementedError("To be implemented in subclasses")

    @property
    @abstractmethod
    def free_parameter_limits(self) -> dict[str, tuple[object, object]]:
        """Free parameters for the problem and their limits

        Returns
        -------
        dict[str, tuple[object, object]]
            Free parameters and their lower and upper limits.
        """
        raise NotImplementedError("To be implemented in subclasses")

    @property
    def free_parameter_distributions(self) -> dict[str, dist.Distribution]:
        """Free parameters for the problem and their distributions

        Returns
        -------
        dict[str, dist.Distribution]
            Free parameters and their distributions.
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

    def calc_distance(self, params):
        """Calculate distance between two points

        The distance measure is defined by the distance attribute.
        It always grows with the Euclidean distance between the points.

        Parameters
        ----------
        params: dict[str, object]
            Pytree with parameter values

        Returns
        -------
        float
            Distance between the two pixels
        """
        k12_kp12 = self.diagonal_indices(params)
        if self.distance == "chord_squared":
            return self.coord.chord_squared(*k12_kp12)
        if self.distance == "minus_cos_arc":
            return -self.coord.cos_arc(*k12_kp12)
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
