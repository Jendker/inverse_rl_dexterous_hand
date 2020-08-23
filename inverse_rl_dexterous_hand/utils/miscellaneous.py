import difflib
from gym import envs
import os


def closest_env_name(env_name):
    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    try:
        return difflib.get_close_matches(env_name, env_ids, n=1)[0]
    except IndexError:
        raise ValueError("None similar environment to {} has been registered".format(env_name)) from None


def job_in_progress(job, training_folder='Runs', logs_dir='logs'):
    try_1 = os.path.join(training_folder, job, logs_dir, "log.csv")
    try_2 = os.path.join(logs_dir, "log.csv")
    return os.path.exists(try_1) or os.path.exists(try_2)
