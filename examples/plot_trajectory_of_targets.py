import jax
import ce3k

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# np.set_printoptions(precision=8, linewidth=150, suppress=True)

rng = jax.random.key(42)
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

# Perform the step transition.
n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)

state = n_state
states = []

for i in range(100 * 10):
    new_key, rng = jax.random.split(rng)
    obs, state, reward, done, info = env.step(new_key, state, 0, env_params)
    states.append(state.target_positions)

for i in range(len(state.target_positions)):
    target = [state[i] for state in states]
    x = [p[0] for p in target]
    y = [p[3] for p in target]
    z = [p[6] for p in target]

    # ax.plot(x, y, z, label=f"Target {i + 1}")
    ax.scatter(x, y, z, label=f"Target {i + 1}", s=1)  # Adjust `s` for marker size


ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_title("Trajectory of 5 Targets over 100k Steps")

plt.show()


print("Done.")