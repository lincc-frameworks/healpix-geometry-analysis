import jax
import jax.numpy as jnp
from healpix_geometry_analysis.problems.tile import TileProblem
from numpyro.infer import MCMC, NUTS


def test_tile_problem_nuts():
    """e2e test for TileProblem"""
    problem = TileProblem.from_order(
        order=4,
        # k_center=1.5,
        # kp_center=14.5,
        k_center=10.5,
        kp_center=15.5,
        direction="m",
        # distance="cos_arc",
        distance="chord_squared",
    )

    kernel = NUTS(problem.model)
    mcmc = MCMC(kernel, num_warmup=0, num_samples=100)
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key)

    samples = mcmc.get_samples()

    arc_distance_deg = jnp.degrees(2.0 * jnp.arcsin(0.5 * jnp.sqrt(samples["distance"])))

    argmin = jnp.argmin(arc_distance_deg)
    min_distance = arc_distance_deg[argmin]
    assert min_distance < problem.coord.grid.average_pixel_size_degree
    # assert min_distance == 0, \
    #     (
    #         f"min_distance = {min_distance:.3f} deg, "
    #         f"average_pixel_size = {problem.coord.grid.average_pixel_size_degree:.3f} deg, "
    #         f"k'1 = {samples['kp1'][argmin]:.3f}, k'2 = {samples['kp2'][argmin]:.3f}"
    #     )

    # import matplotlib.pyplot as plt
    #
    # plt.hist2d(samples["k1"], samples["k2"], bins=15)
    # plt.colorbar()
    # plt.xlabel("k1")
    # plt.ylabel("k2")
    # plt.show()
    # plt.close()
    #
    # plt.hist(arc_distance_deg, bins=100)
    # plt.xlabel("distance, deg")
    # plt.show()
    # plt.close()
