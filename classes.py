import numpy as np
from elastica._linalg import _batch_matvec


class NoForces:
    """
    This is the base class for external forcing boundary conditions applied to rod-like objects.

    Note
    ----
    Every new external forcing class must be derived
    from NoForces class.

    """

    def __init__(self):
        """
        NoForces class does not need any input parameters.
        """
        pass

    def apply_forces(self, system, time: np.float = 0.0):
        """ Apply forces to a rod-like object.

        In NoForces class, this routine simply passes.

        Parameters
        ----------
        system : object
            System that is Rod-like.
        time : float
            The time of simulation.

        Returns
        -------


        """

        pass

    def apply_torques(self, system, time: np.float = 0.0):
        """ Apply torques to a rod-like object.

        In NoForces class, this routine simply passes.

        Parameters
        ----------
        system : object
            System that is Rod-like.
        time : float
            The time of simulation.

        Returns
        -------

        """
        pass


class GravityForces(NoForces):
    """
    This class applies a constant gravitational force to the entire rod.

    Attributes
    ----------
    acc_gravity: numpy.ndarray
        1D (dim) array containing data with 'float' type. Gravitational acceleration vector.

    """

    def __init__(self, acc_gravity=np.array([0.0, -9.80665, 0.0])):
        """

        Parameters
        ----------
        acc_gravity: numpy.ndarray
            1D (dim) array containing data with 'float' type. Gravitational acceleration vector.

        """
        super(GravityForces, self).__init__()
        self.acc_gravity = acc_gravity

    def apply_forces(self, system, time=0.0):
        system.external_forces += np.outer(self.acc_gravity, system.mass)

class TorqueInterval(NoForces):
    def __init__(self, torque: float, direction, time_start: float, time_end: float):
        self.torque = torque
        self.direction = direction
        self.time_start = time_start
        self.time_end = time_end
    def apply_torques(self, system, time: np.float = 0.0):
        if time > self.time_start and time < self.time_end:
            torque_vector = (self.torque * self.direction).reshape(3, 1)
        else:
            torque_vector = np.array([0., 0., 0.]).reshape(3, 1)
        system.external_torques += _batch_matvec(
            system.director_collection, torque_vector / system.n_elems
        )