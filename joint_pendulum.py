import numpy as np
import sys

from collections import defaultdict 

from elastica.wrappers import BaseSystemCollection, Constraints, Forcing, CallBacks, Connections

"""System conditions (i.e rods and boundary conditions)"""
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedRod
from elastica.joint import FreeJoint, HingeJoint, FixedJoint
from elastica.external_forces import GravityForces, UniformForces, EndpointForces, UniformTorques

"""Call back functions - for saving state information during simulation"""
from elastica.callback_functions import CallBackBaseClass

"""Time stepper functions – Currently PositionVerlet is the best default."""
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

from post_processing_joint import plot_video_3D
from classes import TorqueInterval

"""Combine all of the wrappers"""
class Simulator(BaseSystemCollection, Constraints, Forcing, CallBacks, Connections): 
    pass 

"""Initializes simulator"""
hinge_simulator = Simulator()

"""Rod parameter setup: Steel Rod"""
n_elem = 100                            
density_1 = 150                          
nu = 0.1                                
E = 1e6                                 

start_1 = np.zeros((3,))                            
direction = np.array([1.0, 0.0, 0.0])   
normal = np.array([0.0, 0.0, 1.0])      
roll_direction = np.cross(direction, normal)

base_length_1 = 1.0                      
base_radius = 0.025                    
youngs_1 = 1                            
poisson_ratio_1 = .28                     


"""Rod parameter setup: Rubber Rod"""
n_elem = 100                            
density_2 = 500                           
nu = 0.1                                
E = 1e6                                 

start_2 = start_1 + direction * base_length_1                  
direction = np.array([1.0, 0.0, 0.0])   
normal = np.array([0.0, 1.0, 0.0])      

base_length_2 = 2.0                       
base_radius = 0.025                     

youngs_2 = 0.05                           
poisson_ratio_2 = 0.47                    

"""Creates rod object"""
steel_rod = CosseratRod.straight_rod(
    n_elem,
    start_1,
    direction,
    normal,
    base_length_1,
    base_radius,
    density_1,
    nu,
    E,
    youngs_1,
    poisson_ratio_1,
)

rubber_rod = CosseratRod.straight_rod(
    n_elem,
    start_2,
    direction,
    normal,
    base_length_2,
    base_radius,
    density_2,
    nu,
    E,
    youngs_2,
    poisson_ratio_2,
)

"""Adds both rods to simulator"""
hinge_simulator.append(rubber_rod)
hinge_simulator.append(steel_rod)


hinge_simulator.constrain(steel_rod).using(
    OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=None
)

hinge_simulator.connect(
    first_rod=steel_rod, second_rod=rubber_rod, first_connect_idx=-1, second_connect_idx=0
).using(HingeJoint, k=1e5, nu=0, kt=5e3, normal_direction=roll_direction
) # 1e-2


"""Adds gravity force to rod"""
gravity_force = np.array([0.0, 0.0, -9.8])
hinge_simulator.add_forcing_to(rubber_rod).using(
    GravityForces,
    gravity_force,
)

hinge_simulator.add_forcing_to(steel_rod).using(
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
# torque = 3.0
# direction = np.array([0.0, -1.0, 0.0])
# hinge_simulator.add_forcing_to(steel_rod).using(
#     TorqueInterval,
#     torque=torque,
#     direction=direction,
#     time_start=0.0,
#     time_end=0.5
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

rubber_data = defaultdict(list)
steel_data = defaultdict(list)

hinge_simulator.collect_diagnostics(steel_rod).using(
    DynamicPendulum, step_skip=200, callback_params=steel_data
)
hinge_simulator.collect_diagnostics(rubber_rod).using(
    DynamicPendulum, step_skip=200, callback_params=rubber_data
)

print("Callback function added to the simulator")

hinge_simulator.finalize()

final_time = 2.0
dt = 0.0001
total_steps = int(final_time / dt)
print("Total steps to take", total_steps)

timestepper = PositionVerlet()

integrate(timestepper, hinge_simulator, final_time, total_steps)

filename_video = "rubber_rod_torque.mp4"
plot_video_3D(steel_data, rubber_data, video_name=filename_video, fps=125, margins=0.2)




