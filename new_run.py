import sys
import os
import datetime
import copy
import yaml
from evojax.task.base import TaskState
from flax.struct import dataclass
import jax.numpy as jnp

sys.path.append(os.getcwd())
from source.new_natural_env import simulate


def setup_project(config, exp_name):
    """ Prepare the directory of the current project, saved under 'projects'.

    Creates subdirectories and saves the project's configuration.

    Attributes
    ------
    config: dict
        contains configuration for current project

    exp_name: str
        a custom name for the directory

    """
    now = datetime.datetime.now()
    today = str(now.day) + "_" + str(now.month) + "_" + str(now.year)

    top_dir = "projects/"
    project_dir = top_dir + today + "/" + exp_name + "/"


    project_dir += "/trial_" + str(config["trial"])

    if not os.path.exists(project_dir + "/train/data"):
        os.makedirs(project_dir + "/train/data")

    if not os.path.exists(project_dir + "/train/models"):
        os.makedirs(project_dir + "/train/models")

    if not os.path.exists(project_dir + "/train/media"):
        os.makedirs(project_dir + "/train/media")

    if not os.path.exists(project_dir + "/eval/data"):
        os.makedirs(project_dir + "/eval/data")

    if not os.path.exists(project_dir + "/eval/media"):
        os.makedirs(project_dir + "/eval/media")

    print("Saving current simulation under ", project_dir)

    with open(project_dir + "/config.yaml", "w") as f:
        yaml.dump(config, f)

    return project_dir

def simple_run(config):
    """ Run this to reproduce the natural environment described in the paper.
    """
    config["trial"] = 0
    config["agent_view"] = 1
    config["max_time"] = 500
    config["report_freq"] = 1
    config["nb_agents"] = 10000
    config["grid_width"] = 200
    config["grid_length"] = 200
    config["init_food"] = 16000
    config["resources_on"] = True
    config["niches_scale"] = 200
    config["regrowth_scale"] = 0.002
    config["max_age"] = 650
    config["place_agent"] = False
    config["place_resources"] = False
    config["time_reproduce"] = 20
    config["time_death"] = 200
    config["energy_decay"] = 0.03
    config["spontaneous_regrow"] = 0.00005
    config["seed"] = 0
    config["examine_poison"] = False
    config["wall_kill"] = 1

    # project_dir = "."

    # with open(project_dir + "/config.yaml", "r") as f:
    #    config = yaml.safe_load(f)
    project_dir = setup_project(config, "simple_run")

    simulate(project_dir)


if __name__ == "__main__":

    mode = str(sys.argv[1])

    # generic config that you need to change for your simulation
    config = {}

    if mode == "simple":
        simple_run(config)


