import jax.numpy as jnp

SEARCH_OPTIONS = 1
MAX_TRACKERS = 10

AZ_SLICES = 15
EL_SLICES = 5

TMAX = 12345  # Random big number, Do we let these saturate? when?
COSTMAX = 12345  # Random big number, Do we let these saturate? when?

DTYPE_FLOAT = jnp.float16
DTYPE_INT = jnp.int16

POSITION_MINVAL = jnp.array([0, 0, 0], dtype=DTYPE_FLOAT)  # m
POSITION_MAXVAL = jnp.array([200_000, 200_000, 20_000], dtype=DTYPE_FLOAT)  # m
VELOCITY_MINVAL = jnp.array([-1000, -1000, -1000], dtype=DTYPE_FLOAT)  # m/s
VELOCITY_MAXVAL = jnp.array([1000, 1000, 1000], dtype=DTYPE_FLOAT)  # m/s
