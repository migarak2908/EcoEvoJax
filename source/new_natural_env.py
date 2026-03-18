""" Main script for simulating natural environments.
"""

import os
import sys
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
from evojax.util import save_model, load_model
import yaml
import pandas as pd

sys.path.append(os.getcwd())
from source.new_gridworld import Gridworld
from source.utils import VideoWriter


def simulate(project_dir):
    """Simulates the natural environment.

    Parameters
    ----------
    project_dir : str
        Name of project's directory for saving data and models
    """
    with open(project_dir + "/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    key = jax.random.PRNGKey(config["seed"])
    key, env_key = random.split(key)

    env = Gridworld(
        place_agent=config["place_agent"],
        init_food=config["init_food"],
        SX=config["grid_length"],
        SY=config["grid_width"],
        resources_on=config["resources_on"],
        nb_agents=config["nb_agents"],
        regrowth_scale=config["regrowth_scale"],
        niches_scale=config["niches_scale"],
        max_age=config["max_age"],
        time_reproduce=config["time_reproduce"],
        time_death=config["time_death"],
        energy_decay=config["energy_decay"],
        spontaneous_regrow=config["spontaneous_regrow"],
        wall_kill=config["wall_kill"],
        overlap=config["overlap"],
        harm=config["harm"],
        harm_type=config["harm_type"],
        harm_damage=config["harm_damage"],
        selective_reproduction=config["selective_reproduction"],
        num_traits=config["num_traits"],
        trait_mutate_std=config["trait_mutate_std"]
    )

    state = env.reset(env_key)

    video_chunk_size = 100  # Create new video every N steps
    vid = VideoWriter(
        project_dir + "/train/media/steps_0.mp4",
        20.0
    )

    log_data = {
        'step': [],
        'population': [],
        'food': [],
        'action_stay': [],
        'action_move': [],
        'action_reproduce': [],
        'action_attack': []
    }
    agent_log = []
    prev_alive = np.array(state.agents.alive)

    # Main simulation loop
    for steps in range(config["max_time"]):
        state, _, _ = env.step(state)

        current_alive = np.array(state.agents.alive)
        just_died = (prev_alive > 0) & (current_alive == 0)

        if just_died.any():
            dead_indices = np.where(just_died)[0]
            batch_data = {
                  'agent_id': np.array(state.agents.agent_id[dead_indices]),
                  'parent1_id': np.array(state.agents.parent1_id[dead_indices]),
                  'parent2_id': np.array(state.agents.parent2_id[dead_indices]),
                  'death_step': np.full(len(dead_indices), steps),
                  'lifespan': np.array(state.agents.time_alive[dead_indices]),
                  'offspring': np.array(state.agents.nb_offspring[dead_indices]),
                  'food_eaten': np.array(state.agents.nb_food[dead_indices]),
                  'nodes': np.array(state.agents.nodes[dead_indices]),
                  'conns': np.array(state.agents.conns[dead_indices])
              }
            agent_log.append(batch_data)

        prev_alive = current_alive.copy()

        if len(agent_log) >= 1000:
            save_agent_log(agent_log, project_dir, steps)
            agent_log = []




        alive_mask = (state.agents.alive > 0)[:, None]
        alive_actions = state.last_actions * alive_mask
        action_counts = alive_actions.sum(axis=0)

        log_data['step'].append(steps)
        log_data['population'].append(int(state.agents.alive.sum()))
        log_data['food'].append(float(state.state[:, :, 1].sum()))
        log_data['action_stay'].append(int(action_counts[0]))
        log_data['action_move'].append(int(action_counts[1:5].sum()))
        log_data['action_reproduce'].append(int(action_counts[5]))
        if len(action_counts) > 6:
            log_data['action_attack'].append(int(action_counts[6]))
        else:
            log_data['action_attack'].append(0)

        # Report at specified frequency
        if steps % config["report_freq"] == 0:
            population_size = state.agents.alive.sum()
            print(f"Step {steps}, population size {population_size}")
            print(
                f"  Actions: stay={int(action_counts[0])}, move={int(action_counts[1:5].sum())}, reproduce={int(action_counts[5])}",
                end="")
            if len(action_counts) > 6:
                print(f", attack={int(action_counts[6])}")
            else:
                print()

            # Check for extinction
            if population_size == 0:
                print("All agents died")
                break

            pd.DataFrame(log_data).to_csv(project_dir + "/train/data/step_log.csv", index=False)

        if steps % 1 == 0:
            # Add frame to video every step
            rgb_im = state.state[:, :, :3]
            rgb_im = jnp.clip(rgb_im, 0, 1)

            # change color scheme to white green and black
            rgb_im = jnp.clip(rgb_im + jnp.expand_dims(state.state[:, :, 1], axis=-1), 0, 1)
            rgb_im = rgb_im.at[:, :, 1].set(0)
            rgb_im = 1 - rgb_im

            rgb_im = rgb_im - jnp.expand_dims(state.state[:, :, 0], axis=-1)
            rgb_im = np.repeat(rgb_im, 2, axis=0)
            rgb_im = np.repeat(rgb_im, 2, axis=1)

            vid.add(rgb_im)

        # Start new video chunk
        if (steps + 1) % video_chunk_size == 0 and steps + 1 < config["max_time"]:
            vid.close()
            vid = VideoWriter(
                project_dir + f"/train/media/steps_{steps + 1}.mp4",
                20.0
            )

        # Save model at lower frequency
        # if steps % (config["report_freq"] * 10) == 0:
        #     with open(project_dir + "/train/models/step_" + str(steps) + ".pkl", "wb") as f:
        #         pickle.dump({
        #             "nodes": state.agents.nodes,
        #             "conns": state.agents.conns
        #         }, f)

    # Cleanup
    if vid is not None:
        vid.close()

    df = pd.DataFrame(log_data)
    df.to_csv(project_dir + "/train/data/step_log.csv", index=False)

    if agent_log:
        save_agent_log(agent_log, project_dir, "final")


def save_agent_log(agent_log, project_dir, label):
      """Combine batches and save."""
      combined = {
          'agent_id': np.concatenate([b['agent_id'] for b in agent_log]),
          'parent1_id': np.concatenate([b['parent1_id'] for b in agent_log]),
          'parent2_id': np.concatenate([b['parent2_id'] for b in agent_log]),
          'death_step': np.concatenate([b['death_step'] for b in agent_log]),
          'lifespan': np.concatenate([b['lifespan'] for b in agent_log]),
          'offspring': np.concatenate([b['offspring'] for b in agent_log]),
          'food_eaten': np.concatenate([b['food_eaten'] for b in agent_log]),
      }

      # Save stats as CSV (easy analysis)
      df = pd.DataFrame({k: v for k, v in combined.items()
                         if k not in ['nodes', 'conns']})
      df.to_csv(f"{project_dir}/train/data/agent_stats_{label}.csv", index=False)

      # Save networks separately as pickle (large)
      networks = {
          'agent_id': combined['agent_id'],
          'nodes': np.concatenate([b['nodes'] for b in agent_log]),
          'conns': np.concatenate([b['conns'] for b in agent_log])
      }
      with open(f"{project_dir}/train/models/networks_{label}.pkl", "wb") as f:
          pickle.dump(networks, f)



if __name__ == "__main__":
    project_dir = sys.argv[1]
    simulate(project_dir)
