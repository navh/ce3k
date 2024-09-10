#import "@preview/polylux:0.3.1": *
#import themes.university: *

#show: university-theme.with(
  short-author: "Amos  Hebb",
  short-title: "DRDC September Update",
  short-date: "2024-09-10",
  aspect-ratio: "4-3",
  color-a: rgb("#002A5C"),
  color-b: rgb("#008BB0"),
  color-c: rgb("#F2F4F7"),
)

#title-slide(
  authors: ("Amos Hebb\na.hebb@mail.utoronto.ca"),
  title: "DRDC September 2024 Update",
  subtitle: "Initial Simulation Results and Next Steps",
  date: "2024-09-10",
  institution-name: "University of Toronto",
  logo: image("coa.svg", width: 60mm),
)


#slide(title: "Summary")[
  After defending thesis, focused on simulation development.

  In this presentation we will:
  - Discuss the system model.
  - Discuss the simulation environment.
  - Review current state of simulation.
  - Discuss next steps.
]




#slide(title: "System Model", new-section: "Model")[
  - Queue management situation.
  - We consider a *Environment* to be a _Theatre_ and a _Platform_.
  - An *Agent* must select a _task_ from the _queue_.
  - *Theatre* is one or more _targets_.
    - *Target* is an object to detect, kinematics, arrival rate, _etc._

  - *Platform* is one or more _sensors_ and a _processor_.
    - *Sensor* Initially, we an S-Band Multi-Function Radar that can:
      - *search:* detect previously undetected targets, or
      - *track:* update known targets
    - *Processor* processes sensor data and updates the _queue_.

  - *Queue* are a list of _tasks_ the _Agent_ may choose to execute.

]

#slide(title: "Surveillance")[
  - Surveillance region is quantized into a set of $N$ fixed non-overlapping cells
  - The radar uses a number of beams $bold(A) = {1,...,A}$, between which it must divide its time.
    - time allocated to a beam to perform a task denoted as $tau_a$
    - $tau_a$ is fixed, for the S-band radar, at $tau_a = 0.01s$
    - Simulation Arena: $([0,200]"km",[0,200]"km",[0,20]"km")$
    - *Change* Region: Azimuth $[0degree,90degree]$, Elevation $[0degree,30degree]$
    - 15 uniform azimuth cells, 5 unifrom elevation cells.
    - *Change* No range resolution.
  - *Change* We assume a scan duration $Delta T$ of 1 second.
    - Initially we assumed $Delta T = 1$ second
    - Surveillance tasks have a norminal rate of once per $Delta T$
    - Cost-free search coefficient $alpha in (0,1]$.
    - Define $Delta T := (n(bold(A)) times tau_a) / alpha = (75 times 0.01"s")/0.5 = 1.5"s"$
]

#slide(title: "Tracking")[
  - Target motion per Singer trajectory model, we have four targets.
    + Linear: $sigma^2 = 0$
    + Whiplash: $sigma^2 ~U[20m/s^2,35m/s^2], Theta ~U[10s,20s]$
    + Zigzager: $sigma^2 ~U[0m/s^2,5m/s^2], Theta ~U[1s,4s]$
    + Sweeper: $sigma^2 ~U[5m/s^2,20m/s^2], Theta ~U[30s,50s]$

  - Targets are generated at random locations and velocities.

  - In earlier work I've found uniform generated targets make evaluation difficult.
    - Developed a scenario specification format.
]


#slide(title: "Tracking Tasks")[
  - Revisit time $t_r$
    - $t_r = 0.4 ((R_t sigma_theta sqrt(Theta)) / Sigma)^0.4 times U^2.4 / (1+U^2 / 2)$
    - $R_t$ is target range.
    - $Theta$, $Sigma$ are Singer model parameters.
    - We use $U = 0.3$
    - Once a task has terminated, set $t_"start" := t_r$

  - Dwell time
    - $tau_c = tau_c^n times (R_T / R_0)^4 times "SN"_0$
    - $t_c^n = 0.01s, R_T "from target", R_0 = 184"km"$
]


#slide(title: "Gym", new-section: "Simulation")[

  - *Gymnasium* (Gym) is a toolkit for developing and comparing reinforcement learning algorithms.
  - _Game loop_: Time advances until a function releases a sensor, then the agent selects the next function to execute.
  - Environment, theatre and platform.
    - Theatre is designed to support pre-defined scenarios.
    - Platform is designed to support multiple sensors.
  - Observations: The queue.
    - Current idle sensor index as one-hot vector.
    - Single search task, $t_"start"$, $t_"dwell"$, $[t_"elapsed"...]$
    - Each possible track task, $t_"start"$, $t_"dwell"$, $"priority"$ // $t_"elapsed"$, $r_"expected"$
  - Actions: Task index. (Executing a 200ms scheduling frame requires memory)
  - Step: Execute the selected task, calculate reward, update the queue, update the theatre.

]

#slide(title: "JAX")[
  - JAX is a Python library for numerical computing
  - Differentiable rigid body physics engine (BRAX) create and propagate many targets in 3D.
    - Most of the motion and geometry is included in BRAX.
  - Run the simulation in parallel on GPU, much faster.
]

#slide(title: "Demo", new-section: "Demo")[
  Demo the ESTesque simulation.

  `demo_est.py`:
  - Create a theatre and platform.
    - Theatre has 10 targets, all visible, 5 tracked.
    - Platform has 1 sensor.
  - Define the scheduler.

  `est_scheduler.py`:
  - On each step:
    - If buffer has > 0 tasks, execute the first task.
    - Else: While the sum of dwell times is less than 200ms:
      - If all $t_"start" >= tau_a$: a = 0, add a search to buffer.
      - Else: $a = min_{a_i in bold(A)}(t_"start")$, add task earliest start time to buffer.

]

#slide(title: "Next Steps", new-section: "Next Steps")[
  - Implement a multi-sensor platform.
    - Observaiton already supports multiple sensors.
    - Platform definition will need to be updated to support generalized coordinates.
  - Implement a multi-sensor EST.
    - Similar, creates two buffers and re-populates both when either is empty.


]
