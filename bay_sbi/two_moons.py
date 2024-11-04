# coding=utf-8

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for generating two moons simulations."""

import jax.numpy as jnp
import jax

# r_std, r_mean = 0.01, 0.1

def draw_two_moons(rng, theta, r_std=0.01, r_mean = 0.1):
    """Draw two moons following APT paper."""
    rng_a, rng_r = jax.random.split(rng)
    a = jax.random.uniform(rng_a, minval=-jnp.pi/2, maxval=jnp.pi/2)
    r = jax.random.normal(rng_r) * r_std + r_mean
    p = jnp.array([r * jnp.cos(a) + 0.25, r * jnp.sin(a)])
    return p + jnp.array(
        [
            -jnp.abs(jnp.sum(theta))/jnp.sqrt(2),
            (-theta[0] + theta[1])/jnp.sqrt(2)
        ]
    )

def draw_param(rng, x, r_std=0.01, r_mean = 0.1):
        rng_a, rng_r, rng_q = jax.random.split(rng, 3)
        a = jax.random.uniform(rng_a, minval=-jnp.pi/2, maxval=jnp.pi/2)
        r = jax.random.normal(rng_r) * r_std + r_mean
        p = jnp.array([r * jnp.cos(a) + 0.25, r * jnp.sin(a)])

        q_zero = p[0] - x[0]
        q_one = x[1] - p[1]

        flip = jax.random.normal(rng_q) < 0.0
        q_zero = flip * q_zero - (~flip) * q_zero
        q_one = flip * q_one - (~flip) * q_one

        return jnp.array(
            [
                q_zero/jnp.sqrt(2) - q_one/jnp.sqrt(2),
                q_zero/jnp.sqrt(2) + q_one/jnp.sqrt(2)
            ]
        )

def draw_joint_two_moons(rng, theta_min = -1.0, theta_max = 1.0):
    """Return theta and x from two moons simulation."""
    rng_theta, rng_x = jax.random.split(rng)
    theta = jax.random.uniform(
        rng_theta, shape = (2,), minval=theta_min, maxval=theta_max
    )
    return (theta, draw_two_moons(rng_x, theta))