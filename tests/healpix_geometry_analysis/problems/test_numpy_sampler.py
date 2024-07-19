import jax
import jax.numpy as jnp
from healpix_geometry_analysis.geometry.tile import TileGeometry
from healpix_geometry_analysis.problems.numpyro_sampler import NumpyroSamplerProblem
from numpyro.infer import MCMC, NUTS


def test_tile_problem_nuts():
    """e2e test for TileProblem with MCMC sampler"""
    geometry = TileGeometry.from_order(
        order=5,
        k_center=1.5,
        kp_center=14.5,
        direction="p",
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
