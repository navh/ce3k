import jax.numpy as jnp
from jax import random

import gymnasium as gym
from gymnasium import spaces

SEARCH_OPTIONS = 1
SEARCH_ZERO_COST_REVISIT = 1.5
SEARCH_DWELL_DURATION = 0.01
TARGETS = 20
MAX_TRACKERS = 20

AZ_SLICES = 15
EL_SLICES = 5

TMAX = 12345  # Random big number, Do we let these saturate? when?
COSTMAX = 12345  # Random big number, Do we let these saturate? when?

DTYPE_FLOAT = jnp.float32  # TODO: float16 can't represent 200_000m
DTYPE_INT = jnp.int16

# state vector structure [x,vx,ax,y,vy,ay,z,vz,az]
TARGET_MINVAL = jnp.array(
    [0, -1000, -35, 0, -1000, -35, 0, -1000, -35], dtype=DTYPE_FLOAT
)
TARGET_MAXVAL = jnp.array(
    [200_000, 1000, 35, 200_000, 1000, 35, 20_000, 1000, 35], dtype=DTYPE_FLOAT
)


class ce3kEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, theatre=None, platform=None):
        self.theatre = theatre
        self.platform = platform

        self.observation_space = spaces.Dict(
            {
                # "sensor": spaces.Box(0, 1, shape=(1,), dtype=jnp.int16),
                # "sensor": spaces.Discrete(len(self.platform.sensors)),
                "sensor": spaces.Discrete(1),  # TODO: multi-sensor ships
                "search": spaces.Dict(
                    {
                        "t_start": spaces.Box(0, TMAX, shape=(1,), dtype=DTYPE_FLOAT),
                        "t_elapsed": spaces.Box(
                            0,
                            TMAX,
                            shape=(AZ_SLICES * EL_SLICES,),
                            dtype=DTYPE_FLOAT,
                        ),
                    }
                ),
                "trackers": spaces.Dict(
                    {
                        "t_start": spaces.Box(
                            0, TMAX, shape=(MAX_TRACKERS,), dtype=DTYPE_FLOAT
                        ),
                        "t_dwell": spaces.Box(
                            0, TMAX, shape=(MAX_TRACKERS,), dtype=DTYPE_FLOAT
                        ),
                        "on": spaces.Box(0, 1, shape=(MAX_TRACKERS,), dtype=DTYPE_INT),
                        # Ravi mentioned that deadlines were defined a 2x-t_start
                        # "t_deadline": spaces.Box(
                        # 0, TMAX, shape=(MAX_TRACKERS,), dtype=DTYPE_FLOAT
                        # ),
                        # "cost": spaces.Box(
                        # 0, COSTMAX, shape=(MAX_TRACKERS,), dtype=DTYPE_FLOAT
                        # ),
                    }
                ),
            }
        )

        self.action_space = spaces.Discrete(SEARCH_OPTIONS + MAX_TRACKERS)

    def reset(self, seed, options=None):
        assert TARGETS <= MAX_TRACKERS
        super().reset(seed=seed)

        info = {}

        key = random.key(seed)

        # Create a sensor
        sensor = Sensor(0)

        # Create 10 targets with random initial positions
        targets = []
        for i in range(TARGETS):
            key, target_key = random.split(key)
            targets.append(Target(target_key))

        # Create the searcher
        # TODO: This should care about some env_params
        searcher = FunctionSearch()

        # Create 10 trackers
        trackers = []
        for i in range(MAX_TRACKERS):
            trackers.append(FunctionTrack())

        # prime 5 tracker functions with target data
        for i in range(MAX_TRACKERS // 2):
            trackers[i].initialize(
                targets[i]
            )  # TODO: some better association method than positional

        # do some kind of 'full sky raster scan' or something to get things into a plausible state?
        self.theatre = Theatre(targets=targets)
        self.platform = Platform(sensors=[sensor], searcher=searcher, trackers=trackers)
        # observation = self.platform.build_queue()

        observation = self.state_to_observation(self.theatre, self.platform)

        return observation, info

    def state_to_observation(self, theatre, platform):
        return {
            "sensor": 0,
            "search": {
                "t_start": self.platform.searcher.time_desired(),
                "t_elapsed": self.platform.searcher.beams,
            },
            "trackers": {
                "t_start": jnp.array(
                    [tracker.time_desired for tracker in self.platform.trackers],
                    dtype=DTYPE_FLOAT,
                ),
                "t_dwell": jnp.array(
                    [tracker.dwell_time() for tracker in self.platform.trackers],
                    dtype=DTYPE_FLOAT,
                ),
                "on": jnp.array([tracker.on for tracker in self.platform.trackers]),
            },
        }

    def step(self, action):
        info = {}

        # First we do the action
        if action == 0:
            time_delta = self.platform.searcher.execute()
        else:
            time_delta = self.platform.trackers[action - 1].execute()

        info["time_delta"] = time_delta
        # Then we update the theatre and platform
        self.theatre.update(time_delta)
        self.platform.update(time_delta)

        # Then we create a new observation

        observation = self.state_to_observation(self.theatre, self.platform)

        # collect all costs
        cost = 0
        cost += self.platform.searcher.cost
        self.platform.searcher.cost = 0
        for tracker in self.platform.trackers:
            cost += tracker.cost
            tracker.cost = 0  # ew... I don't like monkeying with this over here.
            # TODO: implement some "retrieve_cost" that resets upon retrieval?

        reward = -cost  # slides say divided? I get divide by zero nonsense.
        # Also, I'm only getting reward from the task I just completed.
        # Maybe eventually also take cost from dropping?

        # We never terminate or truncate
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated, info

    # Unpack the action
    # Mark the sensor as busy, ask it for its reward/penalty?
    # Figure out what function will finish next
    # Work out delta t to that function termination
    # Use that delta t to generate a new queue
    # Use that delta t to generate a new theatre


class Platform:
    def __init__(self, sensors=None, searcher=None, trackers=None):
        self.sensors = sensors
        self.searcher = searcher
        self.trackers = trackers

    def update(self, time_delta):
        # Update the searcher
        self.searcher.update(time_delta)
        # Update the trackers
        for i in range(len(self.trackers)):
            self.trackers[i].update(time_delta)


class Sensor:
    def __init__(self, id):
        self.id = id


class FunctionSearch:
    def __init__(self):
        self.beams = jnp.zeros(AZ_SLICES * EL_SLICES, dtype=DTYPE_FLOAT)
        self.last_beam = 0
        self.cost = 0

        # TODO: This should care about what sensor it is executing on def execute(self, sensor):
        # Find the least recently used beam and execute it

    def update(self, t):
        self.beams += t

    def time_desired(self):
        return jnp.max(
            jnp.array([0, SEARCH_ZERO_COST_REVISIT - self.beams[self.last_beam]])
        )

    def execute(self):
        self.cost = jnp.max(
            jnp.array([0, self.beams[self.last_beam] - SEARCH_ZERO_COST_REVISIT])
        )
        # self.beams = self.beams.at[self.last_beam].set(0)
        self.beams = self.beams.at[self.last_beam].set(-SEARCH_DWELL_DURATION)

        self.last_beam = (self.last_beam + 1) % (AZ_SLICES * EL_SLICES)

        # cost = max(0, jnp.max(self.beams) - SEARCH_ZERO_COST_REVISIT)
        # Always looking around for the max is too much work, just search in order
        return SEARCH_DWELL_DURATION


class FunctionTrack:
    def __init__(self):
        self.big_sigma = 0
        self.big_theta = 0
        self.time_desired = 0
        self.on = False
        self.target = None
        self.cost = 0

    def initialize(self, target):
        self.on = True  # TODO: calculate from target?
        # self.big_sigma = target  # TODO: calculate from target?
        self.big_sigma = jnp.linalg.norm(
            jnp.array([target.state[2], target.state[5], target.state[8]])
        )  # average acceleration?
        self.big_theta = 20  # default from matlab code
        self.target = target

    def update(self, t):
        self.time_desired = max(0.0, self.time_desired - t)
        # TODO update self.time_deadline and kill off the tracker if this is exceeded

    def dwell_time(self):
        if not self.on:
            return 0.0
        t_c_super_n = 0.01  # 10ms from the slides
        r_0 = 184_000  # 184km from the slides
        r = self.target.report_spherical_coordinates()[0]
        t_c = t_c_super_n * (r / r_0) ** 4  # Ignore SN_0?
        return t_c

    def execute(self):
        if not self.on:
            return 0.0  # Should these be instant?

        # Calculate Cost
        self.cost += abs(
            self.time_desired
        )  # TODO: weight by priority? TODO: needs delay (need to keep track of how delayed)
        # I think maybe this could be from "deadline", but I think I also need the original "desired" time to dwell.

        # Calculate actual dwell time
        local_dwell = self.dwell_time()

        # Calculate new time desired

        r = self.target.report_spherical_coordinates()[0]
        sigma_theta = 1  # There's a sigma theta here, I'm worried it's some 2d stuff.
        u = 0.3  # Magic number from the slides.
        # This whole t_dwell thing is about to get the ax after the call sep 10th.
        self.time_desired = (
            0.4
            * (r * sigma_theta * jnp.sqrt(self.big_theta) / self.big_sigma) ** 0.4
            * u**2.4
            / (1 + 0.5 * u**2)
        )

        return local_dwell

    def create_queue_entry(self):
        return self.time_desired


class Theatre:
    def __init__(self, targets=None):
        self.targets = targets

    def update(self, t):
        # TODO: https://stonesoup.readthedocs.io/en/latest/stonesoup.models.transition.html#stonesoup.models.transition.linear.Singer
        # So the singer motion model has this crazy transition matrix.
        # I think I need to implement this myself, that said, I don't have a good intuition for its derivation.
        # There seem to be far more, far simpler, transition models out there.
        # It looks like singer is basically just gaussian noise on the acceleration anyway.
        for target in self.targets:
            target.update(t)


class Target:
    def __init__(self, key):
        # Okay so the proposed t_start math is straight up magic.
        # Yeah sure I can figure out the ideal time to start tracking a target if I already knew a target's process noise and maneuvering correlation time.
        # For now... just simple impulse based methods.
        self.state = random.uniform(
            key, (9,), minval=TARGET_MINVAL, maxval=TARGET_MAXVAL, dtype=DTYPE_FLOAT
        )

    def update(self, t):
        # Define the state transition matrix
        A = jnp.array(
            [
                [1, t, 0.5 * t**2, 0, 0, 0, 0, 0, 0],
                [0, 1, t, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, t, 0.5 * t**2, 0, 0, 0],
                [0, 0, 0, 0, 1, t, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, t, 0.5 * t**2],
                [0, 0, 0, 0, 0, 0, 0, 1, t],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.state = A @ self.state

    def report_spherical_coordinates(self):
        return jnp.array(
            [
                jnp.linalg.norm(
                    jnp.array([self.state[0], self.state[3], self.state[6]])
                ),  # range r
                jnp.arccos(
                    self.state[6]
                    / jnp.linalg.norm(
                        jnp.array([self.state[0], self.state[3], self.state[6]])
                    ),
                ),  # elevation theta
                jnp.arctan2(self.state[3], self.state[0]),  # azimuth phi
            ]
        )
