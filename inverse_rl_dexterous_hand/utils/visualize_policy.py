# Based on https://github.com/aravindr93/mjrl.git

import mj_envs
import click 
import os
import pickle
from mjrl.utils.gym_env import GymEnv
import re
import yamlreader

DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python utils/visualize_policy --env_name relocate-v0 --policy policies/relocate-v0.pickle --mode evaluation\n
'''


def get_last_iteration_checkpoint(path):
    files = os.listdir(path)

    def extract_number(f):
        s = re.search("^checkpoint_(\d+)", f)
        return int(s.group(1)) if s else -1, f

    return max(files, key=extract_number)


# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load')
@click.option('--path', type=str, help='absolute path of the policy directory', required=False)
@click.option('--job_name', type=str, help='job name of the polic learned', required=False)
@click.option('--run_no', type=int, default=1)
@click.option('--iteration', '-i', '-it', type=str, help='specify `last` or iteration no. of the policy in path', required=False)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('--horizon', type=float)
def main(env_name, mode, path, iteration, job_name, horizon, run_no):
    env_kwargs = {}
    if path and ('.pickle' in path or 'pkl' in path):
        policy_path = path
    else:
        if job_name:
            path = os.path.join('../inverse_rl_dexterous_hand/training/Runs/', job_name, 'run_' + str(run_no), 'iterations')
        if iteration:
            if iteration == 'last':
                checkpoint_file = get_last_iteration_checkpoint(path)
            else:
                checkpoint_file = "checkpoint_{}.pickle".format(iteration)
            policy_path = os.path.join(path, checkpoint_file)
        else:
            policy_path = os.path.join(path, "best_policy.pickle")
    if env_name is None:
        cfg_path = os.path.join(os.path.dirname(policy_path), "../..", "..", "config.yaml")
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(os.path.dirname(cfg_path), "../..", "config.yaml")
        if not os.path.exists(cfg_path):
            cfg_path = None
        if cfg_path is not None:
            cfg = yamlreader.yaml_load(cfg_path)
            env_name = cfg['env']
            env_kwargs = cfg['env_kwargs']
        else:
            print("Config file not found, cannot infer environment name. Please provide env_name parameter.")
            exit(1)
    e = GymEnv(env_name, **env_kwargs)
    print("Checkpoint path:", policy_path)
    policy = pickle.load(open(policy_path, 'rb'))
    if isinstance(policy, list):
        policy = policy[0]
    # render policy
    if horizon is None:
        horizon = e.horizon
    e.visualize_policy(policy, num_episodes=100, horizon=horizon, mode=mode)

if __name__ == '__main__':
    main()
