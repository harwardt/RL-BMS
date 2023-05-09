# RL-BMS
Code used for the paper 'Lithium-Ion Battery Management System with Reinforcement Learning for Balancing State of Charge and Cell Temperature'. Please read the paper for further information.

# Dependencies
A conda environment with all necessary dependencies can be set up with `environment.yml`.

# Battery Model
The folder `batterymodel` contains `cell.py` for the implementation of a cell and `battery.py` for a battery pack with multiple cells. The implementation follows the modified Shepherd model. The file `battery.py` already includes code necessary for the simulated usage of an active battery management system.

# RL Environment
The reinforcement learning environment is defined in the file `env.py` following the `gym` interface of stable baselines 3.

# Hyperparameter Optimization
The code fore the hyperparameter optimization with HyperOpt is defined in `optimize_parallel.py`. Settings for the training can be made via `config.yml`.

To execute the hyperparameter optimization in parallel the following needs to be done (with the corresponding conda environment activated in all terminals):
1. Start the mongod server in a terminal: 
    mongod --dbpath . --port 1234 --directoryperdb --journal
2. Open a new terminal to run the hyperparameter optimization in with 
    from optimize_parallel import optimize_ppo 
    optimize_ppo() 
The program stops without executing any trials. To run trials stept 3 is necessary.
3. For an execution of n trials in parallel repeat the following n times: 
start a new terminal and run
    PYTHONPATH=. hyperopt-mongo-worker --mongo=localhost:1234/trial_db --poll-interval=0.1