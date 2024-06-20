import dataclasses

import jax
import optax

from healpix_geometry_analysis.problems.base import BaseProblem


@dataclasses.dataclass(kw_only=True)
class OptaxOptimizerProblem(BaseProblem):
    """Description of the optimization problem for optax

    Parameters
    ----------
    geometry : TileGeometry
        Tile geometry object
    """

    def initial_params(self, rng_key: jax.random.PRNGKey) -> dict[str, object]:
        """Sample initial parameter values

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
        for name, distribution in self.geometry.free_parameter_distributions.items():
            random_free_params[name] = distribution.sample(rng_key)
            rng_key = jax.random.split(rng_key)[0]
        all_params = random_free_params | self.geometry.frozen_parameters
        return {name: all_params[name] for name in self.geometry.parameter_names}

    def freeze_optimizer(self, optimizer):
        """Freeze parameters of the Optax optimizer"""
        transforms = {"optimizer": optimizer, "frozen": optax.set_to_zero()}
        param_labels = {frozen: "frozen" for frozen in self.geometry.frozen_parameters} | {
            free: "optimizer" for free in self.geometry.free_parameter_limits
        }
        return optax.multi_transform(transforms, param_labels)

    def optix_loss(self, params):
        """Loss function to minimize with optax"""
        return self.geometry.calc_distance(*(params[name] for name in self.geometry.parameter_names))
