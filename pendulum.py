import numpy as np
import sys

from collections import defaultdict 

from elastica.wrappers import BaseSystemCollection, Constraints, Forcing, CallBacks, Connections

"""System conditions (i.e rods and boundary conditions)"""
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedRod
from elastica.joint import FreeJoint, HingeJoint
from elastica.external_forces import GravityForces, UniformForces, EndpointForces, UniformTorques

"""Call back functions - for saving state information during simulation"""
from elastica.callback_functions import CallBackBaseClass

"""Time stepper functions – Currently PositionVerlet is the best default."""
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

from post_processing import plot_video_3D

"""Combine all of the wrappers"""
class Simulator(BaseSystemCollection, Constraints, Forcing, CallBacks, Connections): 
    pass 

"""Initializes simulator"""
simulator = Simulator()


"""Rod parameter setup: Rubber Rod"""
n_elem = 100                            # Number of elements in rod
density = 500                           # Density of rod (kg/m^3)
nu = 0.1                                # Energy Dissipation of Rod
E = 1e6                                 # Elastic Modulus (Pa)

start = np.zeros((3,))                  # Starting position of first node in rod            
direction = np.array([1.0, 0.0, 0.0])   # Direction the rod extends
normal = np.array([0.0, 0.0, 1.0])      # Normal vector of rod

base_length = 1.0                       # Length of rod (m) 
base_radius = 0.025                     # Radius of rod (m)

youngs = 0.05                           # Stiffness of rod
poisson_ratio = 0.47                    # Expansion or contraction of a material


"""Rod parameter setup: Steel Rod"""
# n_elem = 100                            
# density = 8050                           
# nu = 0.1                                
# E = 1e6                                 

# start = np.zeros((3,))                            
# direction = np.array([1.0, 0.0, 0.0])   
# normal = np.array([0.0, 0.0, 1.0])      

# base_length = 1.0                       
# base_radius = 0.025                    
# youngs = 180                            
# poisson_ratio = .28                     


"""Creates rod object"""
rod = CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    nu,
    E,
    youngs,
    poisson_ratio,
)

"""Adds rod to simulator"""
simulator.append(rod)

# make constrained_director empty
simulator.constrain(rod).using(
    OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

"""Adds gravity force to rod"""
gravity_force = np.array([0.0, 0.0, -9.8])
simulator.add_forcing_to(rod).using(
    GravityForces,
    gravity_force,
)

print("Gravity forces added to the rod")

"""Adds endforce to rod"""
# end_force = np.array([-9.8, 0.0, 0.0])
# simulator.add_forcing_to(rod).using(
#     EndpointForces, 0, end_force, ramp_up_time=5.0
# )

"""Adds torque to rod"""
# torque = 4.5
# direction = np.array([0.0, 0.0, 1.0])
# simulator.add_forcing_to(rod).using(
#     UniformTorques,
#     torque=torque,
#     direction=direction,
# )


class DynamicPendulum(CallBackBaseClass):
    """
    Call back function for pendulum
    """
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        positions = system.position_collection.copy()

        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(positions)

            self.callback_params["x_max"].append(max(positions[0]))
            self.callback_params["x_min"].append(min(positions[0]))

            self.callback_params["y_max"].append(max(positions[1]))
            self.callback_params["y_min"].append(min(positions[1]))

            self.callback_params["z_max"].append(max(positions[2]))
            self.callback_params["z_min"].append(min(positions[2])) 
            return

data_dict = defaultdict(list)
simulator.collect_diagnostics(rod).using(
    DynamicPendulum, step_skip=200, callback_params=data_dict
)

print("Callback function added to the simulator")

simulator.finalize()

final_time = 10.0
dt = 0.0001
total_steps = int(final_time / dt)
print("Total steps to take", total_steps)

timestepper = PositionVerlet()

integrate(timestepper, simulator, final_time, total_steps)

filename_video = "rubber_rod.mp4"
plot_video_3D(data_dict, video_name=filename_video, fps=125, margins=0.2)



