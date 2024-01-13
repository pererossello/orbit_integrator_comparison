import numba 
import numpy as np


@numba.njit
def verlet_step(pos, vel, acc, dt):
    half_vel_next = vel + 0.5 * acc * dt
    pos_next = pos + half_vel_next * dt
    a_next = acc
    vel_next = half_vel_next + 0.5 * a_next * dt
    return pos_next, vel_next

@numba.njit
def rk4_step(pos, vel, acc, dt):
    # First set of calculations (k1)
    k1_vel = acc * dt
    k1_pos = vel * dt

    # Second set of calculations (k2)
    k2_vel = acc * dt
    k2_pos = (vel + 0.5 * k1_vel) * dt

    # Third set of calculations (k3)
    k3_vel = acc * dt
    k3_pos = (vel + 0.5 * k2_vel) * dt

    # Fourth set of calculations (k4)
    k4_vel = acc * dt
    k4_pos = (vel + k3_vel) * dt

    # Combine the results
    pos_next = pos + (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos) / 6
    vel_next = vel + (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel) / 6

    return pos_next, vel_next

@numba.njit
def euler_step(pos, vel, acc, dt):
    # Update velocity
    vel_next = vel + acc * dt

    # Update position
    pos_next = pos + vel * dt

    return pos_next, vel_next

@numba.njit
def leapfrog_step(pos, vel, acc, dt):
    # Perform a half-step velocity update
    vel_half = vel + 0.5 * acc * dt
    # Update position
    pos_next = pos + vel_half * dt
    return pos_next, vel_half

integrator_dic = {
    'verlet': verlet_step,
    'rk4': rk4_step,
    'euler': euler_step,
    'improved_euler': euler_step,
    'leapfrog': leapfrog_step
}