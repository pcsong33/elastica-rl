import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp


_MAX_TENSION = 2.0
_MAX_TORQUE = 3.0
_VALUE_RANGE = 100.0
_V_TH = 0.0000001
_F_TH = 0.0000001


def _get_dists(X):
    """Returns the distances between adjacent body segments."""
    diff = X[1:] - X[:-1]
    return np.linalg.norm(diff, axis=-1)


def _get_iso_coulomb_f(F, X_dot, f_dyn, f_stat_max=None):
    """
    Compute the isotropic coulomb friction.
    Params:
        F: external force (N, 2)
        X_dot: velocity (N, 2)
    Returns:
        Ff: (N, 2)
    """
    if f_stat_max is None:
        f_stat_max = f_dyn
    static = (np.linalg.norm(X_dot, axis=1) < _V_TH).astype(np.float32)

    # Static friction
    F_norm = np.linalg.norm(F, axis=1)
    Ff_stat = -F_norm.clip(max=f_stat_max)[..., None] * F / F_norm[..., None].clip(min=_F_TH)
    
    # Dynamic friction
    v_norm = np.linalg.norm(X_dot, axis=1)
    Ff_dyn = -f_dyn * X_dot / v_norm[..., None].clip(min=_V_TH)

    return static[..., None] * Ff_stat + (1 - static[..., None]) * Ff_dyn


def _aniso_f(cos_theta, f_f, f_b, f_n):
    """
    Compute the max static friction and dynamic friction along a given direction.
    Params:
        cos_theta: relative motion angle with respect to the forward direction.
        f_f: max friction on the forward direction.
        f_b: max friction on the backward direction.
        f_n: max friction on the transverse direction.
    Returns:
        Anisotropic friction
    """    
    is_forward = (cos_theta >= 0).astype(float)
    f = (is_forward * f_f * f_n / np.sqrt(cos_theta**2 * f_n**2 + (1 - cos_theta**2) * f_f**2).clip(min=_F_TH)
         + (1 - is_forward) * f_b * f_n / np.sqrt(cos_theta**2 * f_n**2 + (1 - cos_theta**2) * f_b**2).clip(min=_F_TH))
    return f


def _get_forward_dir(X):
    rs = X[:-1] - X[1:]
    rs = rs / np.linalg.norm(rs, axis=1)[..., None]

    rt = rs[:-1] + rs[1:]
    rt = rt / np.linalg.norm(rt, axis=1)[..., None]
    rt = np.concatenate((rs[0:1], rt, rs[-1:]), 0)

    return rt


def _get_aniso_coulomb_f(F, X, X_dot, f_f, f_b, f_n):
    # forward_dir = _get_forward_dir(X)
    # import pdb; pdb.set_trace()
    
    forward_dir = np.zeros_like(X).astype(float)
    forward_dir[:, 0] = -1.0

    static = (np.linalg.norm(X_dot, axis=1) < _V_TH).astype(np.float32)
    cos_theta = static * np.diag(F.dot(forward_dir.T)) + (1 - static) * np.diag(X_dot.dot(forward_dir.T))
    f = _aniso_f(cos_theta, f_f, f_b, f_n)

    # Static friction
    F_norm = np.linalg.norm(F, axis=1)
    Ff_stat = -F_norm.clip(max=f)[..., None] * F / F_norm[..., None].clip(min=_F_TH)

    # Dynamic friction
    v_norm = np.linalg.norm(X_dot, axis=1)
    Ff_dyn = -f[..., None] * X_dot / v_norm[..., None].clip(min=_V_TH)

    return static[..., None] * Ff_stat + (1 - static[..., None]) * Ff_dyn


def _get_force(X, X_dot, T, Tau, spring_const, L0, friction_type,
               viscosity=5.0,
               f_stat_max=0.5,
               f_dyn=0.5,
               f_f=0.5,
               f_b=None,
               f_n=None):
    """
    Compute the x-y force on each segments
    """
    L = _get_dists(X)
    T_net = T + spring_const * (L - L0) + 5.0 * (L - L0)**3 + 5.0 * (L - L0)**5 

    # Tension force
    Tn = np.concatenate((T_net[..., None] * (X[1:] - X[:-1]) / L[..., None], [[0., 0.]]), 0)  # Tension from the next segm
    Tp = np.concatenate(([[0., 0.]], -Tn[:-1]), 0)  # Tension from the previous segm
    Ft = Tn + Tp

    # Torque force
    fm1x = -(X[:-1] - X[1:])[:-1, 1] * Tau / L[:-1]**2  # Torque force from later seg to earlier seg
    fm1y = (X[:-1] - X[1:])[:-1, 0] * Tau / L[:-1]**2
    fm1 = np.stack((fm1x, fm1y), 1)
    fm1_ = (np.concatenate((fm1, np.zeros((2,2))), 0)
            - np.concatenate((np.zeros((1,2)), fm1, np.zeros((1,2))), 0))

    fm2x = - (X[1:] - X[:-1])[1:, 1] * (-Tau) / L[1:]**2  # Torque force from earlier seg to later seg
    fm2y = (X[1:] - X[:-1])[1:, 0] * (-Tau) / L[1:]**2
    fm2 = np.stack((fm2x, fm2y), 1)
    fm2_ = (np.concatenate((np.zeros((2,2)), fm2), 0)
            - np.concatenate((np.zeros((1,2)), fm2, np.zeros((1,2))), 0))
    Fm = fm1_ + fm2_

    # Friction
    if friction_type == 'viscous':
        Ff = -viscosity * X_dot
    elif friction_type == 'coulomb':
        Ff = _get_iso_coulomb_f(Fm + Ft, X_dot, f_dyn, f_stat_max)
    elif friction_type == 'coulomb_aniso':
        if f_b is None:
            f_b = f_f
        if f_n is None:
            f_n = f_f
        Ff = _get_aniso_coulomb_f(Fm + Ft, X, X_dot, f_f, f_b, f_n)
    else:
        raise ValueError('Invalid friction type {}'.format(friction_type))

    return Ft + Fm + Ff


def _get_omega(X, X_dot):
    """
    Compute the angular velocity between linkages at each body segment.
    Sign is consistent to torque.
    """
    dvs = X_dot[:-1] - X_dot[1:]
    dxs = X[:-1] - X[1:]
    dphidt = dxs[:, 0] / (dxs**2).sum(1) * dvs[:, 1] - dxs[:, 1] / (dxs**2).sum(1) * dvs[:, 0]
    omegas = dphidt[:-1] - dphidt[1:]
    return omegas


def _get_power(X, X_dot, T, Tau):
    """
    Compute the power of actuation force.
    """
    dvs = X_dot[:-1] - X_dot[1:]
    dxs = X[:-1] - X[1:]
    v_rel = np.diag(dvs.dot(dxs.T)) / np.linalg.norm(dxs, axis=1)
    P_T = - (T * v_rel).sum()

    P_Tau = (Tau * _get_omega(X, X_dot)).sum()

    return P_T + P_Tau


class LarvalDrosophilaEnv(gym.Env):

    def __init__(self,
                 n_segm=2,
                 segm_dist=1.0,
                 segm_mass=1.0,
                 time_step=0.01,
                 spring_const=3.0,
                 epoch_steps=5000,
                 friction_type='coulomb',
                 viscosity=5.0,
                 f_stat_max=0.2,
                 f_dyn=0.2,
                 f_f=0.5,
                 f_b=1.0,
                 f_n=3.0,
                 gamma=0.0):
        """
        Params
            n_segm: Number of body segments.
            segm_dist: Distance between adjacent body segments, unstretched spring length.
            segm_mass: Mass of each body segments.
            viscosity: Environment viscosity for friction calculation.
            spring_const: Spring constant between adjacent body segments.
            epoch_steps: Number of steps per epoch, `None` means epoch has no end.
            time_step: Simulation time between two steps.
            gamma: Weight factor of work in reward.
        """
        self._n_segm = n_segm
        self._segm_dist = segm_dist
        self._segm_mass = segm_mass
        self._spring_const = spring_const

        self._friction_type = friction_type
        self._viscosity = viscosity
        self._f_stat_max = f_stat_max
        self._f_dyn = f_dyn
        self._f_f = f_f
        self._f_b = f_b
        self._f_n = f_n
        self._gamma = gamma

        self._epoch_steps = epoch_steps
        self._time_step = time_step

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2*n_segm - 3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-_VALUE_RANGE, high=_VALUE_RANGE,
            shape=(2*n_segm, 2), dtype=np.float32)

        self.reset()

    @property
    def state(self):
        return np.concatenate((self._X, self._X_dot))

    @property
    def reward(self):
        return 0

    @property
    def potential_engergy(self):
        """
        Elastic potential energy of the current state.
        """
        L = _get_dists(self._X)
        return (self._spring_const * (L - self._segm_dist)**2 / 2).sum()

    @property
    def kinetic_energy(self):
        """
        Kinetic energy of the current state.
        """
        return (self._segm_mass * (self._X_dot**2).sum(1) / 2).sum()

    @property
    def work(self):
        """
        Total work done by actuation force.
        """
        return self._work

    @property
    def time(self):
        return self._step_count * self._time_step

    def reset(self):
        xs = np.array([n*self._segm_dist for n in range(self._n_segm)],
                      dtype=np.float32)
        xs -= xs[-1] / 2
        ys = np.zeros_like(xs)
        self._X = np.stack((xs, ys), 1)
        self._X_dot = np.zeros_like(self._X)
        self._work = 0
        self._step_count = 0

        self._work_prev = self._work
        self._X_prev = self._X
        self._X_dot_prev = self._X_dot   

        return self.state   

    def render(self, x_margin=2.0, y_margin=2.0, pixel_size=None):
        fig, ax = plt.subplots()
        for i in range(1, self._X.shape[0]):
            ax.plot(self._X[i-1:i+1, 0], self._X[i-1:i+1,1], 'k')
        ax.scatter(self._X[:, 0], self._X[:, 1])
        ax.scatter(self._X[0, 0], self._X[0, 1], color='r')
        center = self._X.mean(0)
        ax.set_xlim(center[0] - x_margin, center[0] + x_margin)
        ax.set_ylim(center[1] - y_margin, center[1] + y_margin)
        ax.set_aspect('equal')
        ax.grid(True)

        fig.tight_layout()
        fig.canvas.draw()
        plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        if pixel_size:
            pil_img = Image.fromarray(plot)
            pil_img = pil_img.resize((pixel_size[1], pixel_size[0]))
            plot = np.asarray(pil_img)

        return plot

    def _get_init_Y(self):
        """
        Get Y initial values before each step.
        Y ~ (X, X_dot, work)
        """
        return np.concatenate(
            (self._X.reshape(2*self._n_segm),
             self._X_dot.reshape(2*self._n_segm),
             [0]))  # Initial work

    def _Y_to_X(self, Y):
        X = Y[:2*self._n_segm].reshape(self._n_segm, 2)
        X_dot = Y[2*self._n_segm:4*self._n_segm].reshape(self._n_segm, 2)
        work = Y[-1]
        return X, X_dot, work

    def step(self, action):
        T = action[:self._n_segm-1] * _MAX_TENSION  # Tension
        Tau = action[self._n_segm-1:] * _MAX_TORQUE  # Torque

        self._X_prev = self._X  # Keep previous internal state for reward calculation
        self._X_dot_prev = self._X_dot
        self._work_prev = self._work

        def f(_, Y):
            X, X_dot, _ = self._Y_to_X(Y)
            F = _get_force(X, X_dot, T, Tau, self._spring_const, self._segm_dist,
                           friction_type=self._friction_type,
                           viscosity=self._viscosity,
                           f_stat_max=self._f_stat_max,
                           f_dyn=self._f_dyn,
                           f_f=self._f_f,
                           f_b=self._f_b,
                           f_n=self._f_n)
            return np.concatenate(
                (X_dot.reshape(2*self._n_segm),
                 (F / self._segm_mass).reshape(2*self._n_segm),
                 [_get_power(X, X_dot, T, Tau)]))

        # res = solve_ivp(f, (0, self._time_step), self._get_init_Y(), t_eval=[self._time_step], method='RK23')
        # self._X, self._X_dot, work = self._Y_to_X(res.y[:, -1])

        Y = self._get_init_Y()
        # self.X, is the x,y coordinate. 
        self._X, self._X_dot, work = self._Y_to_X(Y + f(0, Y) * self._time_step)
        self._step_count += 1
        self._work += work
        if self._epoch_steps is not None:
            done = self._step_count >= self._epoch_steps
        else:
            done = False        
        info = {'step_work': work}

        return self.state, self.reward, done, info


class Crawler1D(LarvalDrosophilaEnv):

    def __init__(self, *args, **kwargs):
        super(Crawler1D, self).__init__(*args, **kwargs)
        self.observation_space = spaces.Box(  # Relative distance and speed between body segments
            low=-_VALUE_RANGE, high=_VALUE_RANGE,
            shape=(2*(self._n_segm - 1) + self._n_segm,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._n_segm-1,), dtype=np.float32)
    
    @property
    def reward(self):
        """
        Distance travelled along the head direction.
        """
        r_head = self._X_prev[0] - self._X_prev[1]
        u_head = r_head / np.linalg.norm(r_head, 1)
        
        dr_c = self._X.mean(0) - self._X_prev.mean(0)
        dr_c_h = dr_c.dot(u_head)
        
        work = self._work - self._work_prev
        
        return dr_c_h - self._gamma * work
    
    @property
    def state(self):
        """
        Distances between body segments and the speed of each body segment wrt ground
        (relative speed along the body direction.)
        """
        l = _get_dists(self._X)
        dvs = self._X_dot[:-1] - self._X_dot[1:]
        dxs = self._X[:-1] - self._X[1:]
        v_rel = np.diag(dvs.dot(dxs.T)) / np.linalg.norm(dxs, axis=1)
        vs = np.linalg.norm(self._X_dot, axis=-1)
        return np.concatenate((l, v_rel, vs))
    
    def step(self, action):
        torque = np.zeros(self._n_segm - 2).astype(float)  # Only apply longitudinal force, no torque
        full_action = np.concatenate((action, torque))
        return super().step(full_action)
