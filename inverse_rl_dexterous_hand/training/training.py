import os
import yamlreader
import argparse
from inverse_rl_dexterous_hand.training.helper_functions import parse_unknown_args, parse_task, train
import mj_envs


def main():
    parser = argparse.ArgumentParser(description='Policy gradient algorithms with demonstration data.')
    parser.add_argument('--configs', type=str, nargs='+', help='path to additional config parameters')
    parser.add_argument('--continue_task', type=str, default=None, help='task name to continue training')
    args, unknown = parser.parse_known_args()
    custom_parameters = parse_unknown_args(unknown)
    if args.continue_task is not None:
        config_path = os.path.join(args.continue_task, "config.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join('Runs', config_path)
        cfg = yamlreader.yaml_load(config_path)
    else:
        configs = None
        if args.configs is not None:
            configs = []
            for config in args.configs:
                if '.yaml' not in config:
                    config += '.yaml'
                configs.append('../configs/config/' + config)
        cfg = yamlreader.yaml_load(configs, default_path='../configs/default.yaml')
    cfg = yamlreader.data_merge(cfg, custom_parameters)

    # parse multiple runs
    multiple_runs = cfg['runs'] != 'single_run'
    runs_from = cfg['runs'][0] if multiple_runs else 1
    runs_to = cfg['runs'][1] if multiple_runs else 2

    original_seed = cfg['seed']
    for run_no in range(runs_from, runs_to):
        run_seed = original_seed + run_no - 1
        if cfg['algorithm'] == "IRL+DAPG":
            # start IRL
            print("\n----- Starting IRL for DAPG -----\n")
            cfg['algorithm'] = 'IRL'
            if cfg['based_IRL']['IRL_config'] is None:
                IRL_cfg = yamlreader.yaml_load('../configs/config/IRL.yaml')
            else:
                config_path = os.path.join('../configs/config', cfg['based_IRL']['IRL_config'])
                if '.yaml' not in config_path:
                    config_path += '.yaml'
                IRL_cfg = yamlreader.yaml_load(config_path)
            IRL_cfg = yamlreader.data_merge(cfg, IRL_cfg)
            IRL_cfg = yamlreader.data_merge(IRL_cfg, custom_parameters)
            irl_task = parse_task(IRL_cfg)[1]
            train(IRL_cfg, run_no=run_no, multiple_runs=multiple_runs, seed=run_seed)
            # start DAPG based on IRL job
            print("\n----- Starting DAPG based on finished IRL job -----\n")
            cfg['algorithm'] = 'DAPG_based_IRL'
            cfg['based_IRL']['IRL_job'] = irl_task
            cfg['based_IRL']['IRL_run_no'] = run_no if multiple_runs else None
            cfg['use_DAPG'] = True
            train(cfg, run_no=run_no, multiple_runs=multiple_runs, seed=run_seed)
            # make run with initialisation if dump_paths_percentage is set
            start_from_initialisation = IRL_cfg['IRL']['dump_paths_percentage'] is not None
            if start_from_initialisation:
                cfg['based_IRL']['get_paths_for_initialisation'] = True
                train(cfg, run_no=run_no, multiple_runs=multiple_runs, seed=run_seed)
            # reset variables
            cfg['algorithm'] = "IRL+DAPG"
            cfg['based_IRL']['IRL_job'] = None
            cfg['based_IRL']['IRL_run_no'] = None
            if start_from_initialisation:
                del cfg['based_IRL']['get_paths_for_initialisation']
        else:
            train(cfg, run_no=run_no, multiple_runs=multiple_runs, seed=run_seed)


if __name__ == "__main__":
    main()
