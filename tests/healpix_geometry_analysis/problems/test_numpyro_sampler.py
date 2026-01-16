import jax
import jax.numpy as jnp
from healpix_geometry_analysis.coordinates import HealpixCoordinates
from healpix_geometry_analysis.geometry.equatorial import EquatorialGeometry
from healpix_geometry_analysis.geometry.intermediate import IntermediateGeometry
from healpix_geometry_analysis.geometry.polar import PolarGeometry
from healpix_geometry_analysis.geometry.tile import TileGeometry
from healpix_geometry_analysis.problems.numpyro_sampler import NumpyroSamplerProblem
from numpyro.infer import MCMC, NUTS


def test_tile_problem_nuts():
    """e2e test for TileProblem with MCMC sampler"""
    geometry = TileGeometry.from_order(
        order=5,
        k_center=1.5,
        kp_center=14.5,
        direction="m",
        distance="chord_squared",
    )
    problem = NumpyroSamplerProblem(geometry, track_arc_length=True)

    kernel = NUTS(problem.model)
    mcmc = MCMC(kernel, num_warmup=0, num_samples=100)
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key)

    samples = mcmc.get_samples()

    argmin = jnp.argmin(samples["arc_length_degree"])
    min_distance = samples["arc_length_degree"][argmin]
    assert (
        min_distance < problem.geometry.coord.grid.average_pixel_size_degree
    ), f"min_distance samples: {jax.tree.map(lambda x: x[argmin], samples)}"


def test_equatorial_problem_nuts():
    """e2e test for MeridianProblem with MCMC sampler"""
    geometry = EquatorialGeometry.from_order(
        order=2,
        distance="chord_squared",
    )
    problem = NumpyroSamplerProblem(geometry, track_arc_length=True)

    kernel = NUTS(problem.model)
    mcmc = MCMC(kernel, num_warmup=0, num_samples=100)
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key)

    samples = mcmc.get_samples()

    argmin = jnp.argmin(samples["arc_length_degree"])
    min_distance = samples["arc_length_degree"][argmin]
    assert (
        min_distance < problem.geometry.coord.grid.average_pixel_size_degree
    ), f"min_distance samples: {jax.tree.map(lambda x: x[argmin], samples)}"


def test_intermediate_problem_nuts():
    """e2e test for IntermediateProblem with MCMC sampler"""
    min_samples = {}

    for direction in ["p", "m"]:
        geometry = IntermediateGeometry(
            coord=HealpixCoordinates.from_nside(137),
            direction=direction,
            distance="chord_squared",
        )

        problem = NumpyroSamplerProblem(geometry=geometry, track_arc_length=True)

        kernel = NUTS(problem.model)
        mcmc = MCMC(kernel, num_warmup=0, num_samples=200)
        rng_key = jax.random.PRNGKey(int.from_bytes(direction.encode()))
        mcmc.run(rng_key)

        argmin = jnp.argmin(mcmc.get_samples()["arc_length_degree"])
        min_distance = mcmc.get_samples()["arc_length_degree"][argmin]
        # I think it is fine to use argmin indirectly here
        min_sample = jax.tree.map(lambda x: x[argmin], mcmc.get_samples())  # noqa: B023
        assert (
            min_distance < problem.geometry.coord.grid.average_pixel_size_degree
        ), f"min_distance samples: {min_sample}"

        min_samples[direction] = min_sample

    assert (
        min_samples["p"]["arc_length_degree"] > min_samples["m"]["arc_length_degree"]
    ), f"min_samples: {min_samples}"


def test_polar_problem_nuts():
    """e2e test for PolarProblem with MCMC sampler"""
    geometry = PolarGeometry.from_order(
        order=4,
        distance="chord_squared",
        direction="m",
    )
    problem = NumpyroSamplerProblem(geometry, track_arc_length=True)

    kernel = NUTS(problem.model)
    mcmc = MCMC(kernel, num_warmup=0, num_samples=100)
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key)

    samples = mcmc.get_samples()

    argmin = jnp.argmin(samples["arc_length_degree"])
    min_distance = samples["arc_length_degree"][argmin]
    assert (
        min_distance < problem.geometry.coord.grid.average_pixel_size_degree
    ), f"min_distance samples: {jax.tree.map(lambda x: x[argmin], samples)}"
