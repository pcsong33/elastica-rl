import numpy as np
import sys

from collections import defaultdict 

from elastica.wrappers import BaseSystemCollection, Constraints, Forcing, CallBacks, Connections

"""System conditions (i.e rods and boundary conditions)"""
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedRod
from elastica.joint import FreeJoint, HingeJoint
from elastica.external_forces import GravityForces, UniformForces, EndpointForces, UniformTorques, TorqueInterval

"""Call back functions - for saving state information during simulation"""
from elastica.callback_functions import CallBackBaseClass

"""Time stepper functions – Currently PositionVerlet is the best default."""
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

from post_processing import plot_video_3D

import gym
import numpy as np
from gym import spaces


"""Rod parameter setup: Rubber Rod"""
n_elem = 100                            
density = 500                           
nu = 0.1                                
E = 1e6                                 

start = np.zeros((3,))                  
direction = np.array([0.0, 0.0, 1.0])   
normal = np.array([0.0, 1.0, 0.0])      

base_length = 1.0                       
base_radius = 0.025                     

youngs = 0.05                           
poisson_ratio = 0.47                    

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

class CustomEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=10, dtype=np.float16)
    def reset(self):
        pass #todo  