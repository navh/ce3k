#import "@preview/charged-ieee:0.1.2": ieee

// Key Dates
// 2024-11-01: Paper submission deadline
// 2025-01-26: Notification of acceptance
// 2025-03-07: Final paper submission
//

#show: ieee.with(
  title: "Multi-Channel Taskers",
  abstract: [We develop a more comprehensive simulation of a suite of sensors to evaluate radar resource management scheduling algorithms.],
  authors: (
    (
      name: "Amos Hebb",
      department: [Department of Electrical and Computer Engineering],
      organization: [University of Toronto],
      location: [Toronto, Canada],
      email: "a.hebb@mail.utoronto.ca",
    ),
    (
      name: "Sunila Akbar",
      department: [Department of Electrical and Computer Engineering],
      organization: [University of Toronto],
      location: [Toronto, Canada],
      email: "sunila.akbar@utoronto.ca",
    ),
    (
      name: "Raviraj S. Adve",
      department: [Department of Electrical and Computer Engineering],
      organization: [University of Toronto],
      location: [Toronto, Canada],
      email: "rsadve@ece.utoronto.ca",
    ),
    (
      name: "Zhen Ding",
      department: [Radar Sensing and Exploitation Section],
      organization: [Defence Research and Development Canada (DRDC)],
      location: [Ottawa, Canada],
      email: "jack.ding@ecn.forces.gc.ca",
    ),
    (
      name: "Peter W. Moo",
      department: [Radar Sensing and Exploitation Section],
      organization: [Defence Research and Development Canada (DRDC)],
      location: [Ottawa, Canada],
      email: "peter.moo@ecn.forces.gc.ca",
    ),
  ),
  index-terms: ("Cognitive Radar", "Multifunction Radar", "Radar Resource Management"),
  bibliography: bibliography("refs.bib"),
)

= Introduction

Q-Ram _Q-Ram_


= Background

Text @akbar2023transfer and @akbar2024model .

== What does a boat look like?

The main goal is a more comprehensive simulation of a suite of sensors.
Decisions made in this work are inspired by the _River-class destroyer_.

- Aegis Combat System with Canadian Tactical Interface.
- USN Cooperative Engagement Capability (sensor netting).
- Integrated Cyber Defense System.
- OSI Maritime Systems Integrated Bridge and Navigation System.
- L3Harris Internal and External Communication Suite.
- Lockheed Martin AN/SPY-7(V)3 Solid State 3D AESA radar
- MDA Solid State AESA Target Illuminator
- X Band "Surface search antenna" navigation radar.
- S Band navigation radar.
- L3Harris WESCAM Electro-optical and infrared systems.

- NA-30S MK-2 Fire Control System - deal band X-band for search & acquisition, Ka-band for DART guided ammunition.

We do not model underwater sensors, electronic warfare, aviation facilities, or armament systems.

=== Aegis Combat System

Includes an "Aegis Common Source Library" (CSL) and "Open Architecture".
Appears to be closed source.

It manages the detect-to-engage sequuence for anti-air warfare.
The AESA SPY-7 radar autonomously detects and tracks contacts.
Other components classify and identify system tracks.

=== SPY-7
- 1.64 times the detection range of SPY-1
- S-Band radar
- Long range _discrimination_ radar.
- Software wizardry to reduce the resolution difference between it and a dedicated X-band radar.



=== Radar Bands

- L-Band 1-2GHz
- S-Band 2-4GHz
- C-Band 4-8GHz
- X-Band 8-12GHz

=== Functions

- Search
- Confirmation
- Tracking
- Cued search

=== Clutter

- Land
- Sea
- Chaff
- Rain
- Angels (flocks of birds or insects)
- Jammers?


= System Model

The previous work by Akbar _et al._ @akbar2024model did not consider the full system model, but instead model the distributions of task parameters.
We extend this work by considering the full system model, including the radar and the environment.
This is necessary because the radar and the environment are not independent of the task parameters.
By considering the full system model, we get a model of the impact of scheduling decisions on the task parameters for free.

== Radar Model

Surveillance Region is an annulur sector with a radar at the origin, an inner radius of 10 km, an azimuth $theta in [-45 degree, 45 degree]$ and range $r in [10 "km", 184 "km"]$.



= Implementation

In earlier attempts to model radar scheduling problems @hebb2024belief, accelerators and distributed resources#footnote[This research was enabled in part by staff expertise and computing provided by the Digital Research Alliance of Canada (https://alliancecan.ca)] were poorly utilized due to the environment being a bottleneck.
We address this by using JAX @frostig2018jax to write both our environment model and our agent.
This allow us to just-in-time compile (`jit`) the entire training loop using a single-program multiple-data pattern.
Writing within the constraints of JAX gives us the benefit of easily running thousands of seeds in thousands of environments in parallel on a single GPU/TPU, or across multiple GPUs with parallel mapping.

// We take advantage of `equonox` @kidger2021equinox to develop the model.

== Compatability

There is no way to build a complex simulator that produces structured data and get that data into a standard learning algorithm.
PufferLib @suarez2024puffer takes the data and packs it into a flat buffer, this allows a complex simulator to be interacted with like an Atari game.

GPU acceleration for simulation.
Just write for CPU and use PufferLib's emulation layer to run in parallel.

Custom simulation.
Pure C with zero dynamic memory allocations.
c_moba.pyx
Train at over 1,000,000 steps per second.
Centuries of simulated data in a few hours.


= Simulation Results

Performance metrics are presented in three areas: scheduling, detection, and tracking.

== Scheduling

== Detection

== Tracking

= Final Remarks
