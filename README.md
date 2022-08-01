# extremeDiffusion1D
## Description
High performance simulations of random walks in various environments. The costly
functions are written in C++ and then ported to Python using [PyBind11](https://github.com/pybind/pybind11).

## Installation
Installation is automated in bash with the setup.sh script. 

Dependences:  
* [pybind11](http://www.github.com/pybind/pybind11)
*  [npquad](https://github.com/SimonsGlass/numpy_quad)

## Python Iterfaces

### Data Structures

Numerical data is handled using the npquad numpy extension. Floating point data is returned as numpy arrays with ```dtype=np.quad```. Note that quad precision support is limited so downcasting to ```np.float64``` after all calculations are done is recommended. Some helper functions are located in `/pysrc/fileIO.py` and `/pysrc/quadMath.py`.

### Classes

* `pydiffusionCDF.DiffusionTimeCDF`
* `pydiffusionCDF.DiffusionPositionCDF`
* `pydiffusionPDF.DiffusionPDF`

### Examples

```python
from pyDiffusion import DiffusionPDF
from matplotlib import pyplot as plt

# Specify some constants like the number of particles, beta, and number of
# timesteps to evolve the system
nParticles = 1e50
beta = 1
num_of_timesteps = 10_000

# Initialize the system with parameters and other key word arguments
d = DiffusionPDF(
    nParticles,
    beta=beta,
    occupancySize=num_of_timesteps,
    ProbDistFlag=False,
)

# Evolve the system to the specified number of timesteps
maxEdge = []
for _ in range(num_of_timesteps):
    d.iterateTimestep()
    # need to convert index to position in random walk
    maxEdge.append(2*d.getMaxIdx() - d.currentTime)

time = range(1, num_of_timesteps+1)

# Plot the edge over time and save
fig, ax = plt.subplots()
ax.set_xlabel("Time")
ax.set_ylabel("Distance to Center")
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(time, maxEdge)
plt.show()
```
![plot](./examples/MaxEdge.png)
