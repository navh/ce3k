#import "@preview/charged-ieee:0.1.2": ieee

// Key Dates
// 2024-11-01: Paper submission deadline
// 2025-01-26: Notification of acceptance
// 2025-03-07: Final paper submission
//

// Neat stuff
// There's some neat stuff in https://www.canada.ca/content/dam/dnd-mdn/documents/mej/43-150-maritime-engineering-journal-107.pdf#page=15 about how the River-class destroyer is equipped.

#show: ieee.with(
  title: "Evaluate Radar Resource Management Scheduling Algorithms with Comprehensive Simulation",
  abstract: [We develop a more comprehensive simulation of a suite of sensors to evaluate radar resource management scheduling algorithms.],
  authors: (
    (
      name: "Amos Hebb, Sunila Akbar, Raviraj S. Adve",
      department: [Department of Electrical and Computer Engineering],
      organization: [University of Toronto],
      location: [Toronto, Canada],
      email: "a.hebb@mail.utoronto.ca, sunila.akbar@utoronto.ca, rsadve@ece.utoronto.ca",
    ),
    // (
    //   name: "Sunila Akbar",
    //   department: [Department of Electrical and Computer Engineering],
    //   organization: [University of Toronto],
    //   location: [Toronto, Canada],
    //   email: "sunila.akbar@utoronto.ca",
    // ),
    // (
    //   name: "Raviraj S. Adve",
    //   department: [Department of Electrical and Computer Engineering],
    //   organization: [University of Toronto],
    //   location: [Toronto, Canada],
    //   email: "rsadve@ece.utoronto.ca",
    // ),
    (
      name: "Zhen Ding, Peter W. Moo",
      department: [Radar Sensing and Exploitation Section],
      organization: [Defence Research and Development Canada (DRDC)],
      location: [Ottawa, Canada],
      email: "jack.ding@ecn.forces.gc.ca, peter.moo@ecn.forces.gc.ca",
    ),
    // (
    //   name: "Peter W. Moo",
    //   department: [Radar Sensing and Exploitation Section],
    //   organization: [Defence Research and Development Canada (DRDC)],
    //   location: [Ottawa, Canada],
    //   email: "peter.moo@ecn.forces.gc.ca",
    // ),
  ),
  index-terms: ("Cognitive Radar", "Multifunction Radar", "Radar Resource Management"),
  bibliography: bibliography("refs.bib"),
)

= Introduction

Scheduling algorithms for radar resource management are critical for the effective operation of modern radar systems.
Their performance is evaluated using simulations that do not consider the full system model, including the radar and the environment.
We develop a more comprehensive simulation of a suite of sensors to evaluate radar resource management scheduling algorithms.


= Background

Text @akbar2023transfer and @akbar2024model .

Brax @brax2021github

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

MDA Canada has been selected to develop an _X-Band_ Illumination AESA Radar to go below the SPY-7.
#text(
  fill: red,
  [But then other renders omit it, point out a full compartment of active-seeker missiles, and mention that Constellation class omitts illuminators too.],
)

It seems like the brains of the whole operation is
Version 10 of the _Aegis Combat System_ (Aegis).
There is also a

This is an "Aegis Common Source Library" (CSL), with its "Open Architecture", is "Complemented by _Canadian Tactical Interface_ (CTI)"
In this work "Common" means "Bespoke", "Open" means "Closed", and "Complemented by" means "Imcompatible with".

*Aegis* owns:
SPY-7 AESA Radar,
Cooperative Engagement Capability,
Radar Electronic Support Measure (ESM) SEWIP Block 2,
IFF,
Mk-41 VLS,
ESSM Block 2,
SM2,
Tomahawk,
Precision Navigation and Timing (PNT),
Nulka Electronic Warefare Missile Decoy System,
SeaCeptor Close-In Air Defense System (CIADS),
Surface-to-surface naval strike missiles,
and
NATO Link 16/22.

*CTI* owns:
MDA Laser Warning and Countermaesures,
SRD-506 Communication ESM system,
Ultra s2150 Sonar,
Ultras2170 Surface ship torpedo defence system,
Ultra LFAPS-C Towed sonar,
GD Sonobouy processing,
Mk-54 Torpedos,
an "Integrated Communications System" by L3 Harris,
OSI Maritime integrated bridge system,
primary 127mm gun with a NA-30S Mk-2 Fire Control System,
and
secondary 2x30mm Leonardo Lionfish gun system.



Aegis manages the detect-to-engage sequuence for anti-air warfare.
The AESA SPY-7 radar autonomously detects and tracks contacts.
Other components classify and identify system tracks.

SPY-7 S-Band radar (1.64 times the detection range of SPY-1)


== SPY-1 Radar

- 6MW radar
- Search, Tracking, and Missile Guidance
- Track up to 100 targets
- 200km range
- Missile Guidance is RF uplink using SPY-1 radar, then SPG-62 fire-control for splash. Requires scheduling of intercepts.
- River Class uses the SPY-7. S-Band, Active ESA, 3.5MW, 500km range, 1000 targets, 1000km range for missile guidance.

== Something else?

SQL-32(V)6 which is an EW suite is confirmed, but not earlier list.

=== Radar Bands

- L-Band 1-2GHz
- S-Band 2-4GHz
- C-Band 4-8GHz
- X-Band 8-12GHz

=== Functions

- Search
  - Targes use @singer1970estimating.
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
#text(
  fill: red,
  [Ravi/Sunila: Why 184? Also, it's going to be 3D, I think elevation gets $theta$? - Ravi says, this is from BWW. I will revisit BWW. That said, Claire `TAES_CP3_RA3_AH_Aug30.pdf` dials the $sigma_0 = 0.003475m^2$ to get $p_d = 0.75$ at 184km. I don't get it. ],
)


= Implementation

In earlier attempts to model radar scheduling problems @hebb2024belief, accelerators and distributed resources#footnote[This research was enabled in part by staff expertise and computing provided by the Digital Research Alliance of Canada (https://alliancecan.ca)] were poorly utilized due to the environment being a bottleneck.
// TODO: evaluate Cloud TPUs from Google’s TPU Research Cloud (TRC).
// TODO: julia env moves at steps/second.
// TODO: @clairespaper 3m steps in 24hrs despite entire gpu.
We address this by using JAX @frostig2018jax to write both our environment model and our agent.
This allow us to just-in-time compile (`jit`) the entire training loop using a single-program multiple-data pattern.
Writing within the constraints of JAX gives us the benefit of easily running thousands of seeds in thousands of environments in parallel on a single GPU/TPU, or across multiple GPUs with parallel mapping.
// In particular, we use the Brax @brax2021github physics engine.
// While not designed to model sensors at all, being an excellent physics engine, the dynamics of the environment are well captured.

// To define custom models also in JAX, we take advantage of `equonox` @kidger2021equinox to develop the our models.

== Compatability

PufferLib @suarez2024puffer takes the data and packs it into a flat buffer, this allows a complex simulator to be interacted with like an Atari game.
Maybe I'll use this?


= Simulation Results

Performance metrics are presented in three areas: scheduling, detection, and tracking.

== Scheduling

== Detection

== Tracking

= Final Remarks
