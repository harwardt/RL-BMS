from hyperopt import hp,fmin,tpe
from hyperopt.mongoexp import MongoTrials
import os

from gc import callbacks
import os

import stable_baselines3
from env import Env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN,PPO,A2C
from stable_baselines3.common.monitor import Monitor
from custom_callback import EvalCallback_custom
import torch as th
import math
import numpy as np

import yaml


def optimize_ppo():
    space = {
        'learning_rate': hp.uniform('learning_rate', 5e-6, 0.003),
        'n_steps': hp.choice("n_steps", [32, 64, 128, 256, 512, 1024, 2048, 4096]),
        'batch_size': hp.choice("batch_size", [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]),
        'n_epochs': hp.choice("n_epochs", [3, 5, 10, 20, 30]),
        'gamma': hp.choice("gamma", [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
        'gae_lambda': hp.uniform("gae_lambda", 0.9, 1.0),
        'clip_range': hp.choice("clip_range", [0.1, 0.2, 0.3]),
        'ent_coef': hp.uniform("ent_coef", 0, 0.001),
        'vf_coef': hp.choice("vf_coef", [0.5, 1]),
        'max_grad_norm': hp.choice("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]),
        'lr_schedule': hp.choice('lr_schedule', ['linear', 'constant']),
        'hidden_layers_policy_network': hp.choice('hidden_layers_policy_network', [1,2,3,4]),
        'size_hidden_layers_policy_network': hp.choice('size_hidden_layers_policy_network', [32,64,128,256,512,1024]),
        'hidden_layers_value_network': hp.choice('hidden_layers_value_network', [1,2,3,4]),
        'size_hidden_layers_value_network': hp.choice('size_hidden_layers_value_network', [32,64,128,256,512,1024]),
        'ortho_init': hp.choice('ortho_init', [False, True]),
        'activation_fn': hp.choice("activation_fn", ["tanh", "relu"])
    }

    run_trials(space,200)

def run_trials(space,max_trials):
    print('run trials')
    trials = MongoTrials('mongo://localhost:1234/trial_db/jobs', exp_key='exp1')

    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_trials, trials=trials)

def objective(params):

    configfile = open('../config.yml')
    cfgs = yaml.full_load(configfile)
    configfile.close()

    charging = cfgs['charging']
    num_cells = cfgs['num_cells']
    total_timesteps = cfgs['total_timesteps_training']
    length_reference_run = cfgs['length_reference_run']

    print(params)
    print(f'charging: {charging}, num_cells: {num_cells}, total_timesteps_training: {total_timesteps}, length_reference_run: {length_reference_run}')

    th.autograd.set_detect_anomaly(True)

    env = Env(charging = charging, num_cells = num_cells)

    check_env(env, warn=True)

    env = Monitor(env)

    eval_env = Env(charging = charging, num_cells = num_cells, length_reference_run = length_reference_run)

    eval_env = Monitor(eval_env)

    list_check_episodes = [(2**k) * 100 for k in range(6)]
    list_check_episodes = list_check_episodes + [2500 * k for k in range(1,20)]
    list_check_episodes = list_check_episodes + [25000 * k for k in range(1,math.floor(total_timesteps / 25000) + 1)]

    models_per_HP_constellation = 2
    for i in range(models_per_HP_constellation):
        eval_callback = EvalCallback_custom(eval_env, best_model_save_path='./logs'+str(i)+'/',
                                log_path='./logs'+str(i)+'/', eval_freq=list_check_episodes,
                                deterministic=True, render=False, charging = charging, num_cells = num_cells)

        model = set_up_model(params,env)

        model.learn(total_timesteps=total_timesteps, callback=eval_callback)

        model_path_end = './logs'+str(i)+'/end_model'
        model.save(model_path_end)

    model_paths_best = ['./logs' + str(i) + '/best_model' for i in range(models_per_HP_constellation)]

    scores = []

    for i in range(models_per_HP_constellation):
        _ = env.reset()
        model = PPO.load(model_paths_best[i]  + '.zip', env=env)
        scores = scores + [eval_callback.calc_mean_rew(model)]
    
    best = 0
    for i in range(1,models_per_HP_constellation):
        if scores[i]>best:
            best = i

    return np.array(scores).max() * (-1)

def set_up_model(params, env):
    log_path = os.path.join('Training', 'Logs')
    hidden_layers_policy_network = params['hidden_layers_policy_network']
    size_hidden_layers_policy_network = params['size_hidden_layers_policy_network']
    hidden_layers_value_network = params['hidden_layers_value_network']
    size_hidden_layers_value_network = params['size_hidden_layers_value_network']

    net_arch = [dict(pi=([size_hidden_layers_policy_network] * hidden_layers_policy_network),vf=([size_hidden_layers_value_network] * hidden_layers_value_network))]

    activation_fn = {"tanh": th.nn.Tanh, "relu": th.nn.ReLU, "elu": th.nn.ELU, "leaky_relu": th.nn.LeakyReLU}[params['activation_fn']]

    policy_kwargs = dict(net_arch=net_arch,
                    ortho_init=params['ortho_init'],
                    activation_fn=activation_fn)

    batch_size = math.gcd(params['batch_size'],params['n_steps'])

    model = PPO("MlpPolicy",
                env, 
                learning_rate=params['learning_rate'], 
                n_steps=params['n_steps'], 
                batch_size=batch_size, 
                n_epochs=params['n_epochs'], 
                gamma=params['gamma'], 
                gae_lambda=params['gae_lambda'], 
                clip_range=params['clip_range'], 
                ent_coef=params['ent_coef'], 
                vf_coef=params['vf_coef'], 
                max_grad_norm=params['max_grad_norm'],
                policy_kwargs=policy_kwargs,
                verbose=0, 
                tensorboard_log=log_path,
                device='cuda')
    return model