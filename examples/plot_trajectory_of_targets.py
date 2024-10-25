from numpy._core.defchararray import count
import jax
import ce3k
from timeit import default_timer as timer

import matplotlib.pyplot as plt
from matplotlib.cm import rainbow
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# np.set_printoptions(precision=8, linewidth=150, suppress=True)

rng = jax.random.key(0)
rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

print(f"{ce3k.registered_envs}")

# Instantiate environment & its settings.
env, env_params = ce3k.make("dev-v0")

print(env_params)

# Reset the environment.
obs, state = env.reset(key_reset, env_params)

# Sample a random action.
# action = env.action_space(env_params).sample(key_act)
# action = 0
action = 1

jit_step = jax.jit(env.step)

# Perform the step transition.
n_obs, n_state, reward, done, info = jit_step(key_step, state, action, env_params)

state = n_state
target_states = []

STEPS = 100 * 500

start = timer()
xs = []
ys = []
zs = []
# target_states = []

for i in range(STEPS):
    new_key, rng = jax.random.split(rng)
    obs, state, reward, done, info = jit_step(new_key, state, 0, env_params)
    # target_states.append(state.target_positions)
    if i % 200 == 0:
        xs.append([t[0] for t in state.target_positions])
        ys.append([t[3] for t in state.target_positions])
        zs.append([t[6] for t in state.target_positions])
end = timer()

print(f"walltime: {end - start}, Steps: {STEPS}")
print(f"Steps per wallsecond: {STEPS / (end - start)}")
print(f"Wallseconds per megastep: {(end - start) / (STEPS / 1_000_000)}")


x_t = list(map(list, zip(*xs)))
y_t = list(map(list, zip(*ys)))
z_t = list(map(list, zip(*zs)))

count_plotted_targets = 0

for i in range(len(x_t)):
    if sum(x_t[i]) + sum(y_t[i]) + sum(z_t[i]) != 0:
        count_plotted_targets += 1
        ax.scatter(x_t[i], y_t[i], z_t[i], label=f"Target {i}", s=1)

ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_title(f"Trajectory of {count_plotted_targets} Targets over {STEPS} Steps")

plt.show()


print("Done.")
