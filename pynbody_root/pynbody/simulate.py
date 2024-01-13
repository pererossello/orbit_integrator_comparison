import inspect
import copy
import os
import shutil

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy
import sympy
from astropy import units as u
from astropy.constants import G
import numba

from . import integrators
from . import plot_utils as pu

class Body:
    def __init__(self, mass, position, velocity, dtype=np.float64):
        """
        Initializes a Body object.

        Args:
            mass (float or Quantity): The mass of the body.
            position (list or array or Quantity): The initial position of the body as a list of coordinates or Quantity.
            velocity (list or array or Quantity): The initial velocity of the body as a list of coordinates or Quantity.
        """

        if isinstance(position, (list, tuple)) or isinstance(velocity, (list, tuple)):
            position = np.array(position)
            velocity = np.array(velocity)

        if not (position.shape == (3,) and velocity.shape == (3,)):
            raise ValueError("Position and velocity must be three-dimensional.")

        self.mass = dtype(mass)
        self.position = position.astype(dtype)
        self.velocity = velocity.astype(dtype)

        

class NBodySimulation:
    def __init__(self, bodies, e=0):

        self.sim_run = False  # True if simulation has been run
        body_list = copy.deepcopy(bodies)
    

        def is_body(bod): return str(type(bod)) == "<class 'pynbody.simulate.Body'>"
        del bodies
        if not all(is_body(body) for body in body_list):
            raise TypeError(
                "All elements in 'bodies' must be instances of the 'Body' class"
            )
        
        self.bodies = body_list
        self.num_bodies = len(body_list)
        self.masses = np.array([body.mass for body in body_list])

        self.e = e
        self.potential = lambda r: -(r**2 + self.e**2)**(-1/2)
        self.interaction = lambda r: -r * (r**2 + self.e**2)**(-3/2)

        if self.e != 0:
            self.max_acc = 2 / (3*np.sqrt(3) * self.e**2)
        else:
            self.max_acc = np.inf

        self.nonzero_mass_indices = list(np.nonzero(self.masses)[0])
        nonzero_mass_indices_set = set(self.nonzero_mass_indices)
        all_indices = set(range(self.num_bodies))
        self.massless_indices = list(all_indices - nonzero_mass_indices_set)

        self.integrator = 'leapfrog'


    def net_fields(self, compute_energy=False):
        
        fields = np.zeros((self.num_bodies, 3))

        # Compute first the pair interactions between massive bodies
        if compute_energy:
            self.potential_energy[self.step] = 0

        i_already = []
        for i in self.nonzero_mass_indices:
            for j in self.nonzero_mass_indices:  # prevents double counting
                if j in i_already or j == i:
                    continue
                body_i = self.bodies[i]
                body_j = self.bodies[j]
                displacement = body_i.position - body_j.position
                r = np.linalg.norm(displacement)
                force_magnitude = body_i.mass * body_j.mass * self.interaction(r) 
                force_vector = force_magnitude * displacement / r
                fields[i] += force_vector / body_i.mass
                fields[j] -= force_vector / body_j.mass

                if compute_energy:
                    self.potential_energy[self.step] += body_i.mass * body_j.mass * self.potential(r)

            i_already.append(i)

        # Compute the effect of massive bodies on massless bodies
        for i in self.nonzero_mass_indices:
            for j in self.massless_indices:
                body_i = self.bodies[i]
                body_j = self.bodies[j]

                displacement = body_i.position - body_j.position
                r = np.linalg.norm(displacement)
                acc = body_i.mass * self.interaction(r) * displacement / r
                fields[j] -= acc

                if compute_energy:
                    self.potential_energy[self.step] += body_i.mass * self.potential(r)

        return fields


    def update_positions_and_velocities(self, dt, compute_energy=False):

        next_step = integrators.integrator_dic[self.integrator]

        # Update fields (accelerations) based on new positions
        if self.step == 0:
            self.fields = self.net_fields(compute_energy=compute_energy)

        if not self.integrator == 'improved_euler':  
            for i, body in enumerate(self.bodies):
                body.position, body.velocity = next_step(body.position, body.velocity, self.fields[i], dt)
        else:
            fields = copy.deepcopy(self.fields)
            pos_0 = np.array([body.position for body in self.bodies])
            vel_0 = np.array([body.velocity for body in self.bodies])
            for i, body in enumerate(self.bodies):
                body.position, body.velocity = next_step(body.position, body.velocity, self.fields[i], dt)

        self.fields = self.net_fields(compute_energy=compute_energy)

        if self.integrator == 'leapfrog':
            for i, body in enumerate(self.bodies):
                _ , body.velocity = next_step(body.position, body.velocity, self.fields[i], dt)

        if self.integrator == 'improved_euler':
            for i, body in enumerate(self.bodies):
                mean_field = 0.5 * (fields[i] + self.fields[i])
                mean_vel = 0.5 * (vel_0[i] + body.velocity)
                body.position = pos_0[i] + mean_vel * dt
                body.velocity = vel_0[i] + mean_field * dt

            self.fields = self.net_fields(compute_energy=compute_energy)



    def run_simulation(self, duration, time_step, compute_energy=False):

        num_steps = int(duration / time_step)
        num_bodies = len(self.bodies)
        pos_arr = np.zeros((num_steps, 3, num_bodies))
        vel_arr = np.zeros((num_steps, 3, num_bodies))

        if compute_energy:
            self.kinetic_energy = np.zeros(num_steps)
            self.potential_energy = np.zeros(num_steps)
            self.energy = np.zeros(num_steps)
            self.angular_momentum = [0]*num_steps

        for step in range(num_steps):
            self.step = step
            for i, body in enumerate(self.bodies):
                pos_arr[step, :, i] = body.position
                vel_arr[step, :, i] = body.velocity
                if compute_energy:
                    self.kinetic_energy[step] += 0.5 * np.linalg.norm(body.velocity)**2
                    self.angular_momentum[step] += body.mass * np.cross(body.position, body.velocity)

                    if body.mass != 0:
                        self.angular_momentum[step] *= body.mass

            if compute_energy:
                self.angular_momentum[step] = np.linalg.norm(self.angular_momentum[step])


            perc = step/num_steps * 100
            print(f'\r{perc:.2f}%', end='')



            

            self.update_positions_and_velocities(time_step, compute_energy=compute_energy)

        self.positions = pos_arr
        self.velocities = vel_arr
        self.sim_run = True
        self.num_steps = num_steps

        if compute_energy:
            self.energy = self.kinetic_energy + self.potential_energy




    def plot3d_frame(self, frame=0, fig_size=540, frames='all', 
                     azim=0, elev=0, lim=0.75, dpi=300, scatter_size=1):

        fig, axs, fs = pu.initialize_3d(fig_size=fig_size, elev=elev, azim=azim, dpi=dpi, lim=lim)
        facecolor = '#FEFFF5'
        scatter_color = 'w'

        ax = axs[0][0]

        scatter_size = scatter_size*0.25e3/self.num_bodies
        i=frame

        scatter = ax.scatter(self.positions[i,0,:], 
                             self.positions[i,1,:], 
                             self.positions[i,2,:],
                             c='k',
                             s=fs*scatter_size, lw=0.0*fs, 
                             alpha=1)
        
        

    def save_video(self, savefold, fig_size=540, frames='all', 
                   azim=0, elev=0, lim=0.75, dpi=300, 
                   facecolor='#FEFFF5', scatter_color='k',
                   scatter_size=1.5, fps=30,
                   make_plots=True, cmap=None, cmap_val='position'):

        # if savefold is not passed
        if savefold is None:
            raise ValueError('folder where to save the video must be passed!')

        if make_plots:

            fig, axs, fs = pu.initialize_3d(fig_size=fig_size, elev=elev, azim=azim, dpi=dpi, lim=lim, facecolor=facecolor)
            ax = axs[0][0]
            scatter_size = scatter_size*0.25e3/self.num_bodies

            if cmap != None:
                if cmap_val == 'position':
                    color = np.linalg.norm(self.positions[:,:,:], axis=1)
            else:
                c = scatter_color


            inch_size = fig_size/dpi
            sc = 0.15*inch_size
            sh_x = 0.023*inch_size
            sh_y = 0.0275*inch_size

            x0, y0 = sc + sh_x, sc + sh_y
            x1, y1 = inch_size - sc + sh_x, inch_size - sc + sh_y

            if not os.path.exists(savefold):
                os.makedirs(savefold)
            else:
                shutil.rmtree(savefold)
                os.makedirs(savefold)

            step = 1 if frames == 'all' else self.num_steps//frames
            if step == 0:
                step = 1

            for ii, i in enumerate(range(0, self.num_steps, step)):

                if cmap != None:
                    c = color[i,:]

                scatter = ax.scatter(self.positions[i,0,:], 
                                    self.positions[i,1,:], 
                                    self.positions[i,2,:],
                                    c=c,
                                    s=fs*scatter_size, lw=0.0*fs, 
                                    alpha=1)

                fig_name = f'render_{ii:04d}.jpg'
                save_path = savefold + fig_name

                fig.savefig(save_path, dpi=300, 
                            bbox_inches=mpl.transforms.Bbox([[x0, y0], [x1, y1]]),
                            )

                plt.close()

                scatter.remove()

        pu.png_to_gif(savefold, fps=fps)

        return

    
    # def plot_trajectories(self, lim=2):

    #     fig_size, ratio = 540, 3
    #     subplots = (1,3)
    #     Fig = pu.Figure(fig_size=fig_size, ratio=ratio, 
    #                     subplots=subplots)
        
    #     axes = Fig.axes
    #     axes_f = Fig.axes_flat


    #     for j in range(self.num_bodies):
    #         x, y, z = self.positions[:,0,j], self.positions[:,1,j], self.positions[:,2,j]
    #         axes[0][0].plot(pos_arr[j,0,:], pos_arr[j,1,:])
    #         axes[0][1].plot(pos_arr[j,1,:], pos_arr[j,2,:])
    #         axes[0][2].plot(pos_arr[j,0,:], pos_arr[j,2,:])
            
    #     for ax in axes_f:
    #         ax.set_xlim(-lim,lim)
    #         ax.set_ylim(-lim,lim)
    #         ax.set_aspect('equal')
        

    def simple_plot(self, step=1):

        fig_size, ratio = 540, 3
        subplots = (1,3)
        Fig = pu.Figure(fig_size=fig_size, ratio=ratio, 
                        subplots=subplots)

        axes = Fig.axes
        axes_flat = Fig.axes_flat
        axes[0][0].plot(self.positions[:, 0], self.positions[:, 1], alpha=1)
        axes[0][1].plot(self.positions[:, 1], self.positions[:, 2], alpha=1)
        axes[0][2].plot(self.positions[:, 0], self.positions[:, 2], alpha=1)

        for ax in axes_flat:
            ax.set_aspect('equal')


        # for i, sub_dic in self.sim_dic.items():
        #     pos = sub_dic['position']
        #     alpha = 0.2
        #     Fig.axes[0][0].scatter(pos['x'][step], pos['y'][step], color='cyan', alpha=alpha)
        #     Fig.axes[0][1].scatter(pos['x'][step], pos['z'][step], color='cyan', alpha=alpha)
        #     Fig.axes[0][2].scatter(pos['y'][step], pos['z'][step], color='cyan', alpha=alpha)