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
#slide(title: "System Model", new-section: "Model")[
  - We consider an MFR that can:
    - *search:* detect previously undetected targets, or
    - *track:* update known targets
  - Surveillance region is quantized into a set of $N$ fixed non-overlapping cells
  - The radar uses a number of beams $bold(A) = {1,...,A}$, between which it must divide its time.
    - time allocated to a beam to perform a task denoted as $tau_a$
  - Initially, we assume a scan duration $Delta T$ of 1 second.
    - Surveillance tasks have a norminal rate of once per $Delta T$








]


#slide(title: "Simulation", new-section: "Gym")[
  = JAX
  - JAX is a Python library for numerical computing

]

#slide(title: "Demo", new-section: "Demo")[

]