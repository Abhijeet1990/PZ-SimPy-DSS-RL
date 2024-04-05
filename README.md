# PZ-SimPy-DSS-RL
This is a Petting Zoo based multi-agent RL environment for a cyber-physical RL environment DSS-SimPy-RL

Use the `environment.yaml` file to create the conda environment for installing the python packages required in this repository.
Command: `conda env create --file environment.yaml`

**Description of the files within `envs` folder**
- `pz_dss.py` : This is the multi-agent environment for network reconfiguration of a Open-DSS based distribution feeder.
- `pz_dummy.py` : Simple implementation of a rock paper scissor petting zoo based multi-agent environment.
- `pz_simpy_common_3inf.py` : This is the multi-agent environment for network re-routing with each router trained have 3 interfaces.
- `pz_simpy_common_space.py` : This is the multi-agent environment for network re-routing with each router trained equal number of interfaces mainly to ensure uniform action space.
- `pz_simpy_complex.py` : This is the multi-agent environment for network re-routing with each router trained have heterogeneous action spaces.
- `pz_simpy_simple.py` : This is the multi-agent environment for network re-routing with each router trained have heterogeneous action spaces but tested for a smaller network in comparison to `pz_simpy_complex.py`.
- `pz_simpy.py` : It is a broken environment. Kindly do not use!
- The other files are the helper files for the environments.

**Description of the test files**
- `test_agile_rl_pz.py` : This trains a multi-agent environment using MATD3 algorithm for the Simple Speaker Listener environment using Agile RL framework. 
- `test_dummy.py` : Simple training of a rock paper scissor petting zoo based multi-agent environment using stable-baselines.
- `test_env.py` : a dummy test of petting zoo.
- `test_sb.py` : Training of `knights_archers_zombies_v10` environment using stable baselines.
- `test_simpy_common_3inf.py` : Training of `pz_simpy_common_3inf` environment using PPO of stable baselines.
- `test_simpy_complex.py` : Training of `pz_simpy_complex` environment using PPO of stable baselines.
- `train_simpy_complex_agile.py` : Training of `pz_simpy_complex` environment using MATD3 of agile RL.
- `test_simpy_complex_agile.py` : Testing of `pz_simpy_complex` environment using MATD3 of agile RL.
