import os
import joblib
import yamlreader
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.dapg import DAPG
from mjrl.algos.behavior_cloning import BC
from mjrl.utils.train_agent import train_agent
from inverse_rl_dexterous_hand.inverse_rl.irl import irl_training_class
from ruamel.yaml import safe_load, dump
import time as timer
import pickle
from inverse_rl_dexterous_hand.inverse_rl.models.airl_state import AIRL
import tensorflow as tf
from inverse_rl_dexterous_hand.inverse_rl.models.imitation_learning import GAIL, AIRLStateAction
from gym import envs


def get_dict(put_in, key):
    split_names = key.split('.')
    new_dict = {}
    new_dict[split_names[-1]] = put_in
    return new_dict, '.'.join(split_names[:-1])


def parse_unknown_args(unknown):
    assert len(unknown) % 2 == 0, "some custom config keys from arguments have no value assigned"
    import itertools
    custom_parameters = dict(itertools.zip_longest(*[iter(unknown)] * 2, fillvalue=""))
    formatted_custom_parameters = {}
    for key, value in custom_parameters.items():
        new_dict = safe_load(value)
        while key:
            new_dict, key = get_dict(new_dict, key)
        formatted_custom_parameters = yamlreader.data_merge(formatted_custom_parameters, new_dict)
    return formatted_custom_parameters


def parse_task(cfg):
    job_name = cfg['env'] + '_env'
    env_name = cfg['env'] + '-v0'
    if cfg['env'] == 'pen':
        cfg['BC']['epochs'] = 1
    if 'IRL' in cfg and  'visible_indices' in cfg['IRL'] and cfg['IRL']['visible_indices'] is not None:
        if cfg['env'] == 'relocate':
            cfg['IRL']['visible_indices'] = 'slice(-9,None)'
        if cfg['env'] == 'hammer':
            cfg['IRL']['visible_indices'] = 'slice(-19,None)'
        if cfg['env'] == 'pen':
            cfg['IRL']['visible_indices'] = 'slice(-21,None)'
        if cfg['env'] == 'door':
            cfg['IRL']['visible_indices'] = 'slice(-12,None)'
    if 'IRL' in cfg and 'noise_samples' in cfg['IRL'] and cfg['IRL']['noise_samples']:
        cfg['IRL']['noise_samples_generator_args'] = {}
        cfg['IRL']['noise_samples_generator_args']['samples_percent'] = 20
        cfg['IRL']['noise_samples_generator_args']['std_dev_coefficient'] = 1


    cfg['demo_file'] = '../../demonstrations/50_demos_' + cfg['env'] + '-v0.pkl'
    # overwrite job_name if it is DAPG_based_IRL
    if cfg['algorithm'] == 'DAPG_based_IRL':
        actual_IRL_job_name = os.path.split(cfg['based_IRL']['IRL_job'])[1]
        job_name = 'DAPG_based_on_IRL_job_' + actual_IRL_job_name
        if cfg['based_IRL']['IRL_step'] is not None:
            job_name += '_IRL_step_' + str(cfg['based_IRL']['IRL_step'])
    else:
        job_name = cfg['algorithm'] + '_' + job_name

    if cfg['use_DAPG']:
        job_name = 'use_DAPG_' + job_name

    if cfg['train']['augmentation'] > 0:
        job_name = 'direct_learning_augment_' + str(cfg['train']['augmentation']) + '_' + job_name

    if cfg['train']['use_timestamp']:
        job_name = 'use_timestamp_' + job_name

    if 'entropy_weight' in cfg['train'] and cfg['train']['entropy_weight'] > 0:
        job_name = 'entropy_weight_' + str(cfg['train']['entropy_weight']) + '_' + job_name

    if cfg['BC']['epochs'] != 5:
        job_name = 'BC_epochs_' + str(cfg['BC']['epochs']) + '_' + job_name

    if cfg['algorithm'] == 'DAPG_based_IRL':
        if 'get_paths_for_initialisation' in cfg['based_IRL']:
            if cfg['based_IRL']['get_paths_for_initialisation']:
                job_name = 'extended_BC_' + job_name

    if cfg['algorithm'] == 'PPO' or (cfg['algorithm'] == 'IRL' and cfg['IRL']['generator_alg'] == 'PPO'):
        job_name = 'lr_' + str(cfg['PPO']['lr']) + '_' + job_name
        job_name = 'epochs_' + str(cfg['PPO']['epochs']) + '_' + job_name
        job_name = 'PPO_batch_size_' + str(cfg['PPO']['batch_size']) + '_' + job_name

    if cfg['algorithm'] == 'IRL':
        if cfg['IRL']['entropy_weight'] > 0:
            job_name = 'entropy_' + str(cfg['IRL']['entropy_weight']) + '_' + job_name

        if cfg['IRL']['temperature_max'] > 0:
            job_name = 'temperature_max_' + str(cfg['IRL']['temperature_max']) + '_min_' + str(cfg['IRL']['temperature_min']) + '_' + job_name

        if cfg['IRL']['initialization_job']:
            job_name = "init_from_job_" + cfg['IRL']['initialization_job'] + "_" + job_name

        if cfg['IRL']['PCA']['components']:
            scaling_str = 'scaling' if cfg['IRL']['PCA']['scaling'] else 'no_scaling'
            job_name = "FPPCA_" + str(cfg['IRL']['PCA']['components']) + "_" + scaling_str + "_" + job_name

        if cfg['IRL']['mitra_k']:
            job_name = "Mitra_k_" + str(cfg['IRL']['mitra_k']) + "_" + job_name

        if cfg['IRL']['buffer'] is not None:
            job_name = 'buffer_' + cfg['IRL']['buffer'] + '_' + job_name

        if cfg['IRL']['noise_samples_generator_args']['samples_percent'] > 0:
            job_name = 'noise_samples_' + job_name

        if cfg['IRL']['adversarial_samples']['percentage'] > 0:
            job_name = 'adversarial_samples_percentage_' + str(cfg['IRL']['adversarial_samples']['percentage']) + \
                '_adv_range_' + str(cfg['IRL']['adversarial_samples']['range']) + '_adv_alpha_ratio_' + str(cfg['IRL']['adversarial_samples']['alpha_ratio']) + \
                "_" + job_name

        if cfg['IRL']['use_timestamp']:
            job_name = 'use_timestamp_IRL_only_' + job_name

        if cfg['IRL']['discr']['lower_lr_on_main_loop_percentage'] is not None:
            job_name = 'lowering_discrim_lr_dynamically_' + job_name

        if cfg['IRL']['discr']['initialisation_value'] != 0:
            job_name = 'dist_init_value_' + str(cfg['IRL']['discr']['initialisation_value']) + '_' + job_name

        if cfg['IRL']['augmentation']['expert'] > 0:
            job_name = 'augment_expert_' + str(cfg['IRL']['augmentation']['expert']) + '_' + job_name

        if cfg['IRL']['augmentation']['samples'] > 0:
            job_name = 'augment_samples_' + str(cfg['IRL']['augmentation']['samples']) + '_' + job_name

        if cfg['IRL']['augmentation']['lower_sample_augmentation_count'] > 0:
            job_name = 'lower_sample_augmentation_count_' + job_name

        if cfg['IRL']['normalization_lr'] > 0:
            job_name = 'reward_normalization_lr_' + str(cfg['IRL']['normalization_lr']) + '_' + job_name

        if cfg['IRL']['visible_indices']:
            job_name = 'reward_input_masking_' + job_name

        if cfg['IRL']['prefix']:
            job_name = cfg['IRL']['prefix'] + '_' + job_name

        if cfg['IRL']['generator_alg'] == 'PPO':
            job_name = 'PPO_' + job_name

    if cfg['prefix']:
        job_name = cfg['prefix'] + '_' + job_name

    return env_name, job_name


def get_irl_model(env, demos, cfg, seed=None):
    # to check if we should dump paths just to use after training, without buffer for fitting IRL model
    if 'dump_paths_percentage' in cfg['IRL']:
        dump_paths_percentage = cfg['IRL']['dump_paths_percentage']
    else:
        dump_paths_percentage = None
    if seed is None:
        seed = cfg['seed']
    # get model
    return AIRL(env=env, expert_trajs=demos, state_only=cfg['IRL']['discr']['state_only'], fusion=cfg['IRL']['buffer'],
                reward_arch_args=cfg['IRL']['discr']['arch_args'], max_itrs=cfg['IRL']['max_itrs'],
                normalization_learning_rate=cfg['IRL']['normalization_lr'], initialization_job=cfg['IRL']['initialization_job'],
                augment_samples_times=cfg['IRL']['augmentation']['samples'], augment_expert_times=cfg['IRL']['augmentation']['expert'], score_discrim=cfg['IRL']['discr']['score_discrim'], visible_indices=cfg['IRL']['visible_indices'],
                lower_sample_augmentation_count=cfg['IRL']['augmentation']['lower_sample_augmentation_count'], PCA_components=cfg['IRL']['PCA']['components'], PCA_scaling=cfg['IRL']['PCA']['scaling'],
                mitra_k=cfg['IRL']['mitra_k'], seed=seed, dump_paths_percentage=dump_paths_percentage,
                noise_samples_generator_args=cfg['IRL']['noise_samples_generator_args'], adversarial_samples_percentage=cfg['IRL']['adversarial_samples']['percentage'],
                adversarial_samples_range=cfg['IRL']['adversarial_samples']['range'], adversarial_samples_alpha_ratio=cfg['IRL']['adversarial_samples']['alpha_ratio'],
                use_timestamp=cfg['IRL']['use_timestamp'], discr_initialisation_value=cfg['IRL']['discr']['initialisation_value'], alg_use_timestamp=cfg['train']['use_timestamp'])


def add_dumped_paths_for_BC(demo_paths, cfg):
    irl_dumped_paths_path = os.path.join('Runs', cfg['based_IRL']['IRL_job'])
    if cfg['based_IRL']['IRL_run_no'] is not None:
        irl_dumped_paths_path = os.path.join(irl_dumped_paths_path, 'run_' + str(cfg['based_IRL']['IRL_run_no']))
    irl_dumped_paths_path = os.path.join(irl_dumped_paths_path, 'iterations', 'paths_cache')
    paths_folders = [f for f in os.listdir(irl_dumped_paths_path) if os.path.isdir(os.path.join(irl_dumped_paths_path, f))]
    dumped_paths = []
    for paths_folder in paths_folders:
        whole_folder_path = os.path.join(irl_dumped_paths_path, paths_folder)
        paths_files_paths = [os.path.join(whole_folder_path, f) for f in os.listdir(whole_folder_path)
                             if os.path.isfile(os.path.join(whole_folder_path, f)) and f != '.' and f != '..']
        dumped_paths.extend([joblib.load(paths_files_path) for paths_files_path in paths_files_paths])
    bc_demo_paths = demo_paths + dumped_paths
    return bc_demo_paths


def setup_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    return bool(gpus)


def train(cfg, run_no, multiple_runs, seed):
    # ===============================================================================
    # Train Loop
    # ===============================================================================

    gpus_available = setup_gpus()
    env_name, job_name = parse_task(cfg)
    env = GymEnv(env_name, **cfg['env_kwargs'])
    policy = MLP(env.spec, hidden_sizes=tuple(cfg['policy_size']), seed=seed)
    baseline = MLPBaseline(env.spec, reg_coef=1e-3, batch_size=cfg['value_function']['batch_size'],
                           epochs=cfg['value_function']['epochs'], learn_rate=cfg['value_function']['lr'],
                           use_gpu=False)

    # Get demonstration data if necessary and behavior clone
    print("========================================")
    print("Collecting expert demonstrations")
    print("========================================")
    demo_filename = cfg['demo_file']
    if cfg['demo_file'] != None:
        demo_paths = pickle.load(open(demo_filename, 'rb'))
    else:
        demo_paths = None

    if 'demo_file' in cfg['BC'] and cfg['BC']['demo_file'] != 'default':
        bc_demo_file_path = cfg['BC']['demo_file']
        if cfg['train']['use_timestamp']:
            bc_demo_file_path = bc_demo_file_path.replace('v0', 'v0_timestamp_inserted')
        bc_demo_paths = pickle.load(open(bc_demo_file_path, 'rb'))
    else:
        bc_demo_paths = demo_paths
    if cfg['algorithm'] == 'DAPG_based_IRL':
        if 'get_paths_for_initialisation' in cfg['based_IRL']:
            if cfg['based_IRL']['get_paths_for_initialisation']:
                bc_demo_paths = add_dumped_paths_for_BC(demo_paths, cfg)

    ts = timer.time()
    if bc_demo_paths is not None and cfg['BC']['epochs'] > 0:
        print("========================================")
        print("Running BC with expert demonstrations")
        print("========================================")
        bc_agent = BC(bc_demo_paths[25:], policy=policy, epochs=cfg['BC']['epochs'], batch_size=cfg['BC']['batch_size'],
                      lr=cfg['BC']['lr'], loss_type='MSE', set_transforms=True)

        bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")

    if cfg['algorithm'] == 'IRL' or cfg['algorithm'] == 'DAPG_based_IRL':
        IRL_cfg = cfg
        if cfg['algorithm'] == 'DAPG_based_IRL':
            IRL_job_cfg_path = os.path.join("Runs", cfg['based_IRL']['IRL_job'], "config.yaml")
            IRL_cfg = yamlreader.yaml_load(IRL_job_cfg_path)

        irl_model = get_irl_model(env, demo_paths, IRL_cfg, seed)
        if cfg['algorithm'] == 'DAPG_based_IRL':
            full_irl_model_checkpoint_path = os.path.join('Runs', cfg['based_IRL']['IRL_job'])
            if cfg['based_IRL']['IRL_run_no'] is not None:
                full_irl_model_checkpoint_path = os.path.join(full_irl_model_checkpoint_path,
                                                              'run_' + str(cfg['based_IRL']['IRL_run_no']))
            if cfg['based_IRL']['IRL_step'] is not None:
                irl_model.load_iteration(path=full_irl_model_checkpoint_path,
                                         iteration=cfg['based_IRL']['IRL_step'])
            else:
                irl_model.load_last(path=full_irl_model_checkpoint_path)
            irl_model.eval(demo_paths)  # required to load model completely from the given path before changin to different path during training

    if cfg['eval_rollouts'] > 0:
        score = env.evaluate_policy(policy, num_episodes=cfg['eval_rollouts'], mean_action=True)
        print("Score with behavior cloning = %f" % score[0][0])

    if not cfg['use_DAPG']:
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    # ===============================================================================
    # RL Loop
    # ===============================================================================

    irl_kwargs = None
    if cfg['algorithm'] == 'IRL' or cfg['algorithm'] == 'DAPG_based_IRL':
        if cfg['algorithm'] == 'DAPG_based_IRL' or cfg['IRL']['generator_alg'] == 'DAPG':
            generator_algorithm = DAPG
            generator_args = dict(demo_paths=demo_paths, normalized_step_size=cfg['RL']['step_size'],
                                  seed=seed, lam_0=cfg['RL']['lam_0'], lam_1=cfg['RL']['lam_1'], save_logs=cfg['save_logs'],
                                  augmentation=cfg['train']['augmentation'], entropy_weight=cfg['train']['entropy_weight'])
        elif cfg['IRL']['generator_alg'] == 'PPO':
            generator_algorithm = PPO
            generator_args = dict(demo_paths=demo_paths, epochs=cfg['PPO']['epochs'],
                                  mb_size=cfg['PPO']['batch_size'], target_kl_dist=cfg['PPO']['target_kl_dist'],
                                  seed=seed, lam_0=cfg['RL']['lam_0'], lam_1=cfg['RL']['lam_1'],
                                  save_logs=cfg['save_logs'], clip_coef=cfg['PPO']['clip_coef'], learn_rate=cfg['PPO']['lr'],
                                  augmentation=cfg['train']['augmentation'], entropy_weight=cfg['train']['entropy_weight'])
        else:
            raise ValueError("Generator algorithm name", cfg['IRL']['generator_alg'], "not supported")
        irl_class = irl_training_class(generator_algorithm)
        rl_agent = irl_class(env, policy, baseline, train_irl=cfg['algorithm'] != 'DAPG_based_IRL', discr_lr=IRL_cfg['IRL']['discr']['lr'],
                             irl_batch_size=IRL_cfg['IRL']['discr']['batch_size'],
                             lower_lr_on_main_loop_percentage=IRL_cfg['IRL']['discr']['lower_lr_on_main_loop_percentage'],
                             irl_model=irl_model, **generator_args)
        irl_kwargs = dict(policy=dict(min_updates=1, max_updates=IRL_cfg['IRL']['max_gen_updates'] if cfg['algorithm'] != 'DAPG_based_IRL' else 0,
                         steps_till_max=IRL_cfg['IRL']['steps_till_max_gen_updates']))
    elif cfg['algorithm'] == 'DAPG':
        rl_agent = DAPG(env, policy, baseline, demo_paths=demo_paths, normalized_step_size=cfg['RL']['step_size'],
                        lam_0=cfg['RL']['lam_0'], lam_1=cfg['RL']['lam_1'],
                        seed=seed, save_logs=cfg['save_logs'], augmentation=cfg['train']['augmentation'],
                        entropy_weight=cfg['train']['entropy_weight'])
    elif cfg['algorithm'] == 'PPO':
        rl_agent = PPO(env, policy, baseline, demo_paths=demo_paths, epochs=cfg['PPO']['epochs'],
                       mb_size=cfg['PPO']['batch_size'], target_kl_dist=cfg['PPO']['target_kl_dist'],
                       seed=seed, lam_0=cfg['RL']['lam_0'], lam_1=cfg['RL']['lam_1'],
                       save_logs=cfg['save_logs'], clip_coef=cfg['PPO']['clip_coef'],
                       learn_rate=cfg['PPO']['lr'], augmentation=cfg['train']['augmentation'],
                       entropy_weight=cfg['train']['entropy_weight'])
    else:
        raise ValueError("Algorithm name", cfg['algorithm'], "not supported")

    # get IRL model kwargs if doing DAPG based on IRL
    env_kwargs = cfg['env_kwargs']
    if cfg['algorithm'] == 'DAPG_based_IRL':
        rl_agent.irl_model = irl_model

    # dump YAML config file
    job_path = os.path.join("Runs", job_name)
    if not os.path.isdir(job_path):
        os.makedirs(job_path)
    with open(os.path.join(job_path, 'config.yaml'), 'w') as f:
        dump(cfg, f)

    print("========================================")
    print("Starting reinforcement learning phase")
    print("========================================")

    ts = timer.time()
    train_agent(job_name=job_name,
                agent=rl_agent,
                seed=seed,
                niter=cfg['train']['steps'],
                gamma=cfg['train']['gamma'],
                gae_lambda=cfg['train']['gae_lambda'],
                num_cpu=cfg['num_cpu'],
                sample_mode='trajectories',
                num_traj=cfg['train']['num_traj'],
                save_freq=cfg['train']['save_freq'],
                evaluation_rollouts=cfg['eval_rollouts'],
                should_fresh_start=bool(cfg['IRL']['initialization_job']) if cfg['algorithm'] == 'IRL' else False,
                irl_kwargs=irl_kwargs,
                temperature_max=cfg['IRL']['temperature_max'] if cfg['algorithm'] == 'IRL' else 0,
                temperature_min=cfg['IRL']['temperature_min'] if cfg['algorithm'] == 'IRL' else 0,
                plot_keys=cfg['plot_keys'],
                run_no=run_no if multiple_runs else None,
                env_kwargs=env_kwargs,
                fixed_evaluation_init_states=cfg['fixed_evaluation_init_states'])
    print("time taken = %f" % (timer.time()-ts))
