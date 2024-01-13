# Visualizing comparison of different orbital mechanics integrators 

Generates videos of orbits and energy in a N-two-body simulation. 

<!-- ![Video GIF Placeholder](figures/NFW_k/video_20.gif)
![Alt text](figures/NFW_k/video_20.gif "NFW") -->

![1e5 particles with Euler and Leapfrog time-step](example.mp4)


## Structure
- `pynbody_root/pynbody/integrators.py`: 
  - Different numerical integrator functions defined

- `pynbody_root/pynbody/simulate.py`: 
  - Classes for N-body problem simulation

- `pynbody_root/pynbody/plot_utils.py`: 
  - Plotting utility functions

- `test_symplectic.ipynb`: 
  - Main notebook to run simulation and get the video output. 

## Dependencies
This project requires the following Python packages:
- [numpy]
- [matplotlib]
- [numba]
- [PIL]
- [ffmpeg]

