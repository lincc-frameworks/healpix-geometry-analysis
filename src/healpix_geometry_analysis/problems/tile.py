import dataclasses
from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax

from healpix_geometry_analysis.coordinates import HealpixCoordinates

DIRECTIONS = ["p", "m"]
DIRECTION_T = Literal[*DIRECTIONS]

DISTANCE = ["chord_squared", "cos_arc"]
DISTANCE_T = Literal[*DISTANCE]


@dataclasses.dataclass(kw_only=True)
class TileProblem:
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
    distance : {"chord_squared", "cos_arc"}
        Distance function to use:
        - "chord_squared" for squared chord distance in the unit sphere
        - "cos_arc" for cosine of the great circle arc distance
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
    """Distance function to use, "chord_squared" or "cos_arc\""""

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
        distance : {"chord_squared", "cos_arc"}
            Distance function to use:
            - "chord_squared" for squared chord distance in the unit sphere
            - "cos_arc" for cosine of the great circle arc distance

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

    def numpyro_side1(self):
        """Get k & k' numpyro samples for the first side of the tile

        It is NE for "p" direction and NW for "m" direction
        """
        return self._numpyro_side(1)

    def numpyro_side2(self):
        """Get k & k' numpyro samples for the second side of the tile

        It is SW for "p" direction and SE for "m" direction
        """
        return self._numpyro_side(2)

    def _numpyro_side(self, index: Literal[1, 2]) -> tuple[object, object]:
        k_name = f"k{index}"
        kp_name = f"kp{index}"
        if self.direction == "p":
            k = numpyro.sample(k_name, self.free_parameter_distributions[k_name])
            kp = self.frozen_parameters[kp_name]
        elif self.direction == "m":
            k = self.frozen_parameters[k_name]
            kp = numpyro.sample(kp_name, self.free_parameter_distributions[kp_name])
        else:
            raise ValueError(f"Invalid direction: {self.direction}, must be one of {DIRECTIONS}")
        return k, kp

    def numpyro_model(self):
        """Numpyro model tp maximize"""
        k1, kp1 = self.numpyro_side1()
        k2, kp2 = self.numpyro_side2()

        if self.distance == "chord_squared":
            distance = self.coord.chord_squared(k1, kp1, k2, kp2)
            # Use negative distance to minimize it
            numpyro.factor("target", -distance)
        elif self.distance == "cos_arc":
            distance = self.coord.cos_arc(k1, kp1, k2, kp2)
            # Use positive cosine to maximize it and minimize distance
            numpyro.factor("target", distance)
        else:
            raise ValueError(f"Invalid distance: {self.distance}, must be one of {DISTANCE}")

        numpyro.deterministic("distance", distance)

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
            limits = (self.k_center - self.delta, self.k_center + self.delta)
            return dict.fromkeys(["k1", "k2"], limits)
        if self.direction == "m":
            limits = (self.kp_center - self.delta, self.kp_center + self.delta)
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

    def params(self, rng_key: jax.random.PRNGKey) -> dict[str, object]:
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

    def freeze_optax_optimizer(self, optimizer):
        """Freeze parameters of the optimizer"""
        transforms = {"optimizer": optimizer, "frozen": optax.set_to_zero()}
        param_labels = {frozen: "frozen" for frozen in self.frozen_parameters} | {
            free: "optimizer" for free in self.free_parameter_limits
        }
        return optax.multi_transform(transforms, param_labels)

    def optix_side1(self, params):
        """Get k & k' from the optimizer parameters for the first side of the tile"""
        return self._optix_side(params, 1)

    def optix_side2(self, params):
        """Get k & k' from the optimizer parameters for the second side of the tile"""
        return self._optix_side(params, 2)

    def _optix_side(self, params, index: Literal[1, 2]) -> tuple[object, object]:
        k_name = f"k{index}"
        kp_name = f"kp{index}"
        if self.direction == "p":
            k = params[k_name]
            kp = self.frozen_parameters[kp_name]
        elif self.direction == "m":
            k = self.frozen_parameters[k_name]
            kp = params[kp_name]
        else:
            raise ValueError(f"Invalid direction: {self.direction}, must be one of {DIRECTIONS}")
        return k, kp

    def optix_loss(self, params):
        """Loss function to minimize with optix"""
        k1, kp1 = self.optix_side1(params)
        k2, kp2 = self.optix_side2(params)

        if self.distance == "chord_squared":
            return self.coord.chord_squared(k1, kp1, k2, kp2)
        if self.distance == "cos_arc":
            # Use negative cosine to maximize it and minimize distance
            return -self.coord.cos_arc(k1, kp1, k2, kp2)
        raise ValueError(f"Invalid distance: {self.distance}, must be one of {DISTANCE}")
