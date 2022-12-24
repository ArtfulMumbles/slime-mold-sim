# Slime Mold Simulation

This program is a simulation of slime mold behavior, inspired by 
[Characteristics of pattern formation and evolution in approximations
of physarum transport networks](https://uwe-repository.worktribe.com/output/980579)
by Jeff Jones and the work of [Sebastian Lague](https://sebastian.itch.io) on his
YouTube channel. Other learning resources include:

- [Intelligent Particle Simulations in Python, Slime Molds and Their Beauty](https://medium.com/geekculture/intelligent-particle-simulations-in-python-slime-molds-and-their-beauty-c9527200f997) by Eric Lastname
- [Physarum Simulation](https://www.michaelfogleman.com/projects/physarum/) by Michael Fogleman
- The work of [Sage Jensen](https://cargocollective.com/sagejenson/physarum)

## Installation

The code is written in Python 3.11 and uses the following libraries: `numpy`, `random`, `glob`, `time`, and `cv2`.

## Usage

The simulation has many parameters that can be modified. It is important that you create a **`/results`**
directory such that simulation results can be saved. When calling the `simulation()` function, using 
parameter `True` will output a map of agents while `False` will output a map of trails.

## Project Status
The following features are slotted to be added in the future:
1. Parallelization through GPU with `ModernGL` and `ComputeShader`.
2. Increased sensor size.
3. Migrate to compiled language, hopefully **`Rust`**

## License

This project is licensed under the MIT License.
