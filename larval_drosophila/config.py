import os

from yacs.config import CfgNode


_PROJ_HOME = ('output')


def get_default():
    """Get default config."""

    config = CfgNode()

    # Experiment
    config.run_dir = os.path.join(_PROJ_HOME, 'test_run')
    config.load_path = ''  # Only used for continuous training
    config.debug = False
    config.seed = 0
    
    # Environment
    config.env = CfgNode()
    config.env.name = 'crawler_1d'

    config.env.crawler_1d = CfgNode()
    config.env.crawler_1d.n_segm = 2
    config.env.crawler_1d.segm_dist = 1.0
    config.env.crawler_1d.segm_mass = 1.0
    config.env.crawler_1d.time_step = 0.01
    config.env.crawler_1d.spring_const = 3.0
    config.env.crawler_1d.epoch_steps = 1000
    config.env.crawler_1d.friction_type = 'coulomb_aniso'
    config.env.crawler_1d.f_f = 0.001
    config.env.crawler_1d.f_b = 0.5
    config.env.crawler_1d.f_n = 0.5
    config.env.crawler_1d.gamma = 5.0

    # Policy
    config.model = CfgNode()
    config.model.name = 'a2c'
    config.model.policy = 'MlpPolicy'
    
    config.model.a2c = CfgNode()
    config.model.a2c.gamma = 0.99
    config.model.a2c.n_steps = 1
    config.model.a2c.vf_coef = 0.25
    config.model.a2c.ent_coef = 0.01
    config.model.a2c.learning_rate = 0.001
    # config.model.a2c.learning_rate = 1e-4

    config.model.ppo = CfgNode()
    config.model.ddpg = CfgNode()
    
    return config


def set_params(config, file_path=None, list_opt=None):
    """Set config parameters with config file and options.
    Option list (usually from cammand line) has the highest
    overwrite priority.
    """
    if file_path:
        # if list_opt is None or 'run_dir' not in list_opt[::2]:
        #     raise ValueError('Must specify new run directory.')
        print('- Import config from file {}.'.format(file_path))
        config.merge_from_file(file_path)
    if list_opt:
        print('- Overwrite config params {}.'.format(str(list_opt[::2])))
        config.merge_from_list(list_opt)
    return config


def freeze(config, save_file=False):
    """Freeze configuration and save to file (optional)."""
    config.freeze()
    if save_file:
        if not os.path.isdir(config.run_dir):
            os.makedirs(config.run_dir)
        with open(os.path.join(config.run_dir, 'config.yaml'), 'w') as fout:
            fout.write(config.dump())
