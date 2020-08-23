# Based on https://github.com/aravindr93/mjrl.git

import mj_envs
import click
import gym
import pickle

DESC = '''
Helper script to visualize demonstrations.\n
USAGE:\n
    Visualizes demonstrations on the env\n
    $ python utils/visualize_demos --env_name relocate-v0\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required=True)
@click.option('--demo', type=str, help='demo file type', required=False)
@click.option('--demo_path', '--path', type=str, help='demo file path '
                                            '(if specified, parameter --demo is ignored)', required=False)
def main(env_name, demo, demo_path):
    if env_name == "":
        print("Unknown env.")
        return
    if demo_path is None:
        if env_name == 'relocate-v0':
            demo_path = './demonstrations/' + env_name + '_demos.pickle'
        elif env_name == 'relocateSR-v0':
            if demo == 'merged':
                demo_path = 'inverse_rl_dexterous_hand/inverse_rl_dexterous_hand/record/performed_demonstrations/merged/merged.pkl'
            elif demo == 'replayed' or demo == 'replay':
                demo_path = 'inverse_rl_dexterous_hand/inverse_rl_dexterous_hand/record/demonstrations/relocateSR-v0.pkl'
            else:
                print('Demo parameter for relocateSR not specified or invalid. Exiting')
                exit(1)
    print("Demo path:", demo_path)
    demos = pickle.load(open(demo_path, 'rb'))
    # render demonstrations
    demo_playback(env_name, demos)

def demo_playback(env_name, demo_paths):
    e = gym.make(env_name)
    e.reset()
    for path in demo_paths:
        e.env.set_env_state(path['init_state_dict'])
        actions = path['actions']
        for t in range(actions.shape[0]):
            out = e.step(actions[t])
            e.env.mj_render()

if __name__ == '__main__':
    main()
