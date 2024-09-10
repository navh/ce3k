import gymnasium as gym

env = gym.make("ce3k:navh/ce3kEnv-v0")
observation, info = env.reset(seed=42)


def earliest_start_time_policy(observation):
    if not any(observation["trackers"]["on"]):
        return 0  # no trackers are on, so start a search

    earliest = 999999
    index_of_earliest = -1
    for i in range(len(observation["trackers"]["t_start"])):
        if (
            observation["trackers"]["on"][i]
            and observation["trackers"]["t_start"][i] < earliest
        ):
            earliest = observation["trackers"]["t_start"][i]
            index_of_earliest = i

    if observation["search"]["t_start"] < earliest:
        return 0  # search is earliest
    if earliest >= 0.01:
        return 0  # Room for a search before the earliest tracker's desired start time
    return index_of_earliest + 1  # +1 because 0 is reserved for search


# Start by cycling everything at least once
for _ in range(len(observation["search"]["t_elapsed"])):
    observation, reward, terminated, truncated, info = env.step(0)
for i in range(len(observation["trackers"]["t_start"])):
    observation, reward, terminated, truncated, info = env.step(i + 1)

action_history = []
for _ in range(1000):
    action = earliest_start_time_policy(observation)
    action_history.append(action)
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation["trackers"])
    print(f"a = {action}, t = {info['time_delta']}, cost = {reward}")

print(action_history)
