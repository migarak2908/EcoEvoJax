# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import Tuple
from PIL import Image
from PIL import ImageDraw
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass
import math

import sys

from reprod.adapted_neat.genome import DefaultGenome

sys.path.insert(0, '/Users/migarakumarasinghe/Documents/Project/EcoEvoJax/reprod')

AGENT_VIEW = 1

from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask


@dataclass
class AgentStates(object):
    posx: jnp.uint16
    posy: jnp.uint16
    nodes: jnp.ndarray
    conns: jnp.ndarray
    seqs: jnp.ndarray
    u_conns: jnp.ndarray
    energy: jnp.ndarray
    time_good_level: jnp.uint16
    time_alive: jnp.uint16
    time_under_level: jnp.uint16
    alive: jnp.int8
    nb_food: jnp.ndarray
    nb_offspring: jnp.uint16


@dataclass
class State(TaskState):
    obs: jnp.int8
    last_actions: jnp.int8
    rewards: jnp.int8
    state: jnp.int8
    agents: AgentStates
    steps: jnp.int32
    key: jnp.ndarray


def get_ob(state: jnp.ndarray, pos_x: jnp.int32, pos_y: jnp.int32) -> jnp.ndarray:
    obs = (jax.lax.dynamic_slice(jnp.pad(state, ((AGENT_VIEW, AGENT_VIEW), (AGENT_VIEW, AGENT_VIEW), (0, 0))),
                                 (pos_x - AGENT_VIEW + AGENT_VIEW, pos_y - AGENT_VIEW + AGENT_VIEW, 0),
                                 (2 * AGENT_VIEW + 1, 2 * AGENT_VIEW + 1, 3)))
    # obs=jnp.ravel(state)

    return obs


def get_init_state_fn(key: jnp.ndarray, SX, SY, posx, posy, pos_food_x, pos_food_y, niches_scale=200, resources_on=True) -> jnp.ndarray:
    grid = jnp.zeros((SX, SY, 4))
    grid = grid.at[posx, posy, 0].add(1)
    grid = grid.at[posx[:5], posy[:5], 0].set(0)

    if resources_on:
        grid = grid.at[pos_food_x, pos_food_y, 1].set(1)

    new_array = jnp.clip(
        np.asarray([(math.pow(niches_scale, el) - 1) / (niches_scale - 1) for el in np.arange(0, SX) / SX]), 0,
        1)

    for col in range(SY - 1):
        new_col = jnp.clip(
            np.asarray([(math.pow(niches_scale, el) - 1) / (niches_scale - 1) for el in np.arange(0, SX) / SX]), 0, 1)

        new_array = jnp.append(new_array, new_col)

    new_array = jnp.transpose(jnp.reshape(new_array, (SY, SX)))
    grid = grid.at[:, :, 3].set(new_array)
    # grid=grid.at[600:700,:300,3].set(0)
    # grid=grid.at[:,:,3].set(5)

    grid = grid.at[0, :, 2].set(1)
    grid = grid.at[-1, :, 2].set(1)
    grid = grid.at[:, 0, 2].set(1)
    grid = grid.at[:, -1, 2].set(1)

    return (grid)


get_obs_vector = jax.vmap(get_ob, in_axes=(None, 0, 0), out_axes=0)


class Gridworld(VectorizedTask):
    """gridworld task."""

    def __init__(self,
                 nb_agents: int = 100,
                 init_food=16000,
                 SX=300,
                 SY=100,
                 reproduction_on=True,
                 resources_on=True,
                 place_resources=False,
                 place_agent=False,
                 test: bool = False,
                 energy_decay=0.05,
                 max_age: int = 1000,
                 time_reproduce: int = 150,
                 time_death: int = 40,
                 max_ener=3.,
                 regrowth_scale=0.002,
                 niches_scale=200,
                 spontaneous_regrow=1 / 200000,
                 wall_kill=True,
                 ):
        self.obs_shape = (AGENT_VIEW, AGENT_VIEW, 3)
        # self.obs_shape=11*5*4
        self.act_shape = tuple([5, ])
        self.test = test
        self.nb_agents = nb_agents
        self.SX = SX
        self.SY = SY
        self.energy_decay = energy_decay
        self.max_age = max_age
        self.time_reproduce = time_reproduce
        self.time_death = time_death
        self.max_ener = max_ener

        self.regrowth_scale = regrowth_scale
        self.niches_scale = niches_scale
        self.spontaneous_regrow = spontaneous_regrow
        self.place_agent = place_agent
        self.resources_on = resources_on
        self.place_resources = place_resources
        self.reproduction_on = reproduction_on

        self.genome = DefaultGenome(
            num_inputs=(2 * AGENT_VIEW + 1) ** 2 * 3,
            num_outputs=6,
            max_nodes=50,
            max_conns=1000
        )

        def reset_fn(key):

            if self.place_agent:
                next_key, key = random.split(key)
                posx = random.randint(next_key, (self.nb_agents,), int(2 / 5 * SX), int(3 / 5 * SX))
                next_key, key = random.split(key)
                posy = random.randint(next_key, (self.nb_agents,), int(2 / 5 * SY), int(3 / 5 * SY))
                next_key, key = random.split(key)

            else:
                next_key, key = random.split(key)
                posx = random.randint(next_key, (self.nb_agents,), 1, (SX - 1))
                next_key, key = random.split(key)
                posy = random.randint(next_key, (self.nb_agents,), 1, (SY - 1))
                next_key, key = random.split(key)

            if resources_on:
                if self.place_resources:
                    # lab environments have a custom location of resources
                    N = 5  # minimum distance from agents
                    N_wall = 5  # minimum distance from wall

                    pos_food_x = jnp.concatenate(
                        (random.randint(next_key, (int(init_food / 4),), int(1 / 2 * SX) + N, (SX - 1 - N_wall)),
                         random.randint(next_key, (int(init_food / 4),), N_wall, int(1 / 2 * SX) - N),
                         random.randint(next_key, (int(init_food / 4),), 1 + N_wall, (SX - 1 - N_wall)),
                         random.randint(next_key, (int(init_food / 4),), 1 + N_wall, (SX - 1 - N_wall))))

                    next_key, key = random.split(key)
                    pos_food_y = jnp.concatenate(
                        (random.randint(next_key, (int(init_food / 4),), 1 + N_wall, SY - 1 - N_wall),
                         random.randint(next_key, (int(init_food / 4),), 1 + N_wall, SY - 1 - N_wall),
                         random.randint(next_key, (int(init_food / 4),), int(1 / 2 * SY) + N,
                                        (SY - 1 - N_wall)),
                         random.randint(next_key, (int(init_food / 4),), N_wall, int(1 / 2 * SY) - N)))
                    next_key, key = random.split(key)

                else:
                    pos_food_x = random.randint(next_key, (init_food,), 1, (SX - 1))
                    next_key, key = random.split(key)
                    pos_food_y = random.randint(next_key, (init_food,), 1, (SY - 1))
                    next_key, key = random.split(key)
                grid = get_init_state_fn(key, SX, SY, posx, posy, pos_food_x, pos_food_y, niches_scale, resources_on=resources_on)
            else:
                pos_food_x = 0
                pos_food_y = 0
                grid = get_init_state_fn(key, SX, SY, posx, posy, pos_food_x, pos_food_y, niches_scale, resources_on=resources_on)

            next_key, key = random.split(key, 2)


            initialize_keys = jax.random.split(key, self.nb_agents)
            nodes, conns = jax.vmap(self.genome.initialize, in_axes=(None, 0))(
                None, initialize_keys
                )
            seqs, _, _, u_conns = jax.vmap(self.genome.transform, in_axes=(None, 0, 0))(None, nodes, conns)


            agents = AgentStates(posx=posx, posy=posy, nodes=nodes, conns=conns, seqs=seqs, u_conns=u_conns,
                                 energy=self.max_ener * jnp.ones((self.nb_agents,)).at[0:5].set(0),
                                 time_good_level=jnp.zeros((self.nb_agents,), dtype=jnp.uint16),
                                 time_alive=jnp.zeros((self.nb_agents,), dtype=jnp.uint16),
                                 time_under_level=jnp.zeros((self.nb_agents,), dtype=jnp.uint16),
                                 alive=jnp.ones((self.nb_agents,), dtype=jnp.uint16).at[0:2 * self.nb_agents // 3].set(
                                     0),
                                 nb_food=jnp.zeros((self.nb_agents,)),
                                 nb_offspring=jnp.zeros((self.nb_agents,), dtype=jnp.uint16)
                                 )

            return State(state=grid, obs=get_obs_vector(grid, posx, posy), last_actions=jnp.zeros((self.nb_agents, 6)),
                         rewards=jnp.zeros((self.nb_agents, 1)), agents=agents,
                         steps=jnp.zeros((), dtype=int), key=next_key)

        self._reset_fn = jax.jit(reset_fn)

        def mating_reproduce(action_int, posx, posy, nodes, conns, seqs, u_conns, energy, time_good_level, key, time_alive, alive, nb_food,
                      nb_offspring):

            """
            1. Create dead mask as before
            2. Choose 5 empty spots
            3. Identify valid parents - parents who are occupying the same space and both pressed action 6 (this might be tough) and greater than time_good_level
            4. Select one of the parents - use the standard asexual reproduction mechanism
            """


            num_agents = action_int.shape[0]
            max_pairs = self.nb_agents // 2

            next_key, key = random.split(key)


            reproducers = (action_int[:, 5] == 1) & (time_good_level > self.time_reproduce) & (alive > 0)
            same_x = posx[:, None] == posx[None, :]
            same_y = posy[:, None] == posy[None, :]

            same_pos = same_x & same_y
            same_pos = same_pos & (~jnp.eye(num_agents, dtype=bool))

            valid_pairs = same_pos & (reproducers[:, None] & reproducers[None, :])

            valid_pairs = jnp.triu(valid_pairs, k=1)

            # parent_i, parent_j = jnp.where(valid_pairs, size=max_pairs, fill=-1)
            parent_i, parent_j = jnp.where(valid_pairs, size=max_pairs)

            # num_pairs = (parent_i >= 0).sum()
            # parent_i = jnp.where(parent_i < 0, 0, parent_i)  # safe dummy index
            # parent_j = jnp.where(parent_j < 0, 0, parent_j)
            num_pairs = valid_pairs.sum()


            dead = (alive == 0)
            # empty_spots = jnp.where(dead, size=self.nb_agents, fill=-1)[0]
            empty_spots = jnp.where(dead, size=self.nb_agents)[0]
            num_empty = dead.sum()

            # num_offspring = jnp.minimum(num_pairs, dead.sum())  # max offspring = # dead
            num_offspring = jnp.minimum(num_pairs, num_empty)


            parent1_nodes = nodes[parent_i]
            parent2_nodes = nodes[parent_j]
            parent1_conns = conns[parent_i]
            parent2_conns = conns[parent_j]

            next_key, key = random.split(key)
            crossover_keys = random.split(next_key, max_pairs)

            offspring_nodes, offspring_conns = jax.vmap(
                self.genome.execute_crossover, in_axes=(None, 0, 0, 0, 0, 0)
            )(None, crossover_keys, parent1_nodes, parent1_conns, parent2_nodes, parent2_conns)

            next_key, key = random.split(key)
            mutation_keys = random.split(next_key, max_pairs)

            next_key, key = random.split(key)
            new_node_keys = jax.random.randint(next_key, (max_pairs,), 1000, 100000)
            next_key, key = random.split(key)
            new_conn_keys = jax.random.randint(next_key, (max_pairs, 3), 1000, 100000)

            offspring_nodes, offspring_conns = jax.vmap(
                self.genome.execute_mutation, in_axes=(None, 0, 0, 0, 0, 0)
            )(None, mutation_keys, offspring_nodes, offspring_conns, new_node_keys, new_conn_keys)

            offspring_seqs, _, _, offspring_u_conns = jax.vmap(
                self.genome.transform, in_axes=(None, 0, 0)
            )(None, offspring_nodes, offspring_conns)

            place_mask = jnp.arange(max_pairs) < num_offspring
            node_mask = place_mask[:, None, None]
            conn_mask = place_mask[:, None, None]
            seq_mask = place_mask[:, None]
            u_conn_mask = place_mask[:, None, None]

            target_spots = empty_spots[:max_pairs]
            target_spots = jnp.where(target_spots < 0, 0, target_spots)

            nodes = nodes.at[target_spots].set(
                jnp.where(node_mask, offspring_nodes, nodes[target_spots])
            )
            conns = conns.at[target_spots].set(
                jnp.where(conn_mask, offspring_conns, conns[target_spots])
            )
            seqs = seqs.at[target_spots].set(
                jnp.where(seq_mask, offspring_seqs, seqs[target_spots])
            )
            u_conns = u_conns.at[target_spots].set(
                jnp.where(u_conn_mask, offspring_u_conns, u_conns[target_spots])
            )

            # Update positions for offspring
            posx = posx.at[target_spots].set(
                jnp.where(place_mask, posx[parent_i], posx[target_spots])
            )
            posy = posy.at[target_spots].set(
                jnp.where(place_mask, posy[parent_i], posy[target_spots])
            )

            energy = energy.at[target_spots].set(
                jnp.where(place_mask, self.max_ener, energy[target_spots])
            )
            alive = alive.at[target_spots].set(
                jnp.where(place_mask, 1, alive[target_spots])
            )
            time_alive = time_alive.at[target_spots].set(
                jnp.where(place_mask, 0, time_alive[target_spots])
            )
            time_good_level = time_good_level.at[target_spots].set(
                jnp.where(place_mask, 0, time_good_level[target_spots])
            )
            nb_food = nb_food.at[target_spots].set(
                jnp.where(place_mask, 0, nb_food[target_spots])
            )
            nb_offspring = nb_offspring.at[target_spots].set(
                jnp.where(place_mask, 0, nb_offspring[target_spots])
            )

            nb_offspring = nb_offspring.at[parent_i].add(jnp.where(place_mask, 1, 0))
            nb_offspring = nb_offspring.at[parent_j].add(jnp.where(place_mask, 1, 0))

            return (
            nodes, conns, seqs, u_conns, posx, posy, energy, time_good_level, time_alive, alive, nb_food, nb_offspring)


        def step_fn(state):
            key = state.key
            next_key, key = random.split(key)

            # TODO Write batched action selection for heterogenous agents
            # model selection of action
            obs_flat = state.obs.reshape(self.nb_agents, -1)
            per_agent_transformed = (
                state.agents.seqs,
                state.agents.nodes,
                state.agents.conns,
                state.agents.u_conns
            )


            actions_logits = jax.vmap(
                self.genome.forward, in_axes=(None, (0, 0, 0, 0), 0)
            )(None, per_agent_transformed, obs_flat)

            actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logits * 50, axis=-1), 6)

            grid = state.state
            energy = state.agents.energy
            alive = state.agents.alive

            # move agent
            action_int = actions.astype(jnp.int32)
            posx = state.agents.posx - action_int[:, 1] + action_int[:, 3]
            posy = state.agents.posy - action_int[:, 2] + action_int[:, 4]

            # wall
            hit_wall = state.state[posx, posy, 2] > 0

            if (wall_kill):
                alive = jnp.where(hit_wall, 0, alive)
            posx = jnp.where(hit_wall, state.agents.posx, posx)
            posy = jnp.where(hit_wall, state.agents.posy, posy)

            posx = jnp.clip(posx, 0, SX - 1)
            posy = jnp.clip(posy, 0, SY - 1)
            grid = grid.at[state.agents.posx, state.agents.posy, 0].set(0)
            # add only the alive
            grid = grid.at[posx, posy, 0].add(1 * (alive > 0))

            ### collect food

            rewards = (alive > 0) * (grid[posx, posy, 1] > 0) * (1 / (grid[posx, posy, 0] + 1e-10))
            grid = grid.at[posx, posy, 1].add(-1 * (alive > 0))
            grid = grid.at[:, :, 1].set(jnp.clip(grid[:, :, 1], 0, 1))

            nb_food = state.agents.nb_food + rewards

            # regrow

            num_neighbs = jax.scipy.signal.convolve2d(grid[:, :, 1], jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
                                                      mode="same")
            scale = grid[:, :, 3]
            scale_constant = regrowth_scale
            next_key, key = random.split(state.key)

            if scale_constant:

                num_neighbs = jnp.where(num_neighbs == 0, 0, num_neighbs)
                num_neighbs = jnp.where(num_neighbs == 1, 0.01 / 5, num_neighbs)
                num_neighbs = jnp.where(num_neighbs == 2, 0.01 / scale_constant, num_neighbs)
                num_neighbs = jnp.where(num_neighbs == 3, 0.05 / scale_constant, num_neighbs)
                num_neighbs = jnp.where(num_neighbs > 3, 0, num_neighbs)
                num_neighbs = jnp.multiply(num_neighbs, scale)
                num_neighbs = jnp.where(num_neighbs > 0, num_neighbs, 0)
                # num_neighbs = num_neighbs + self.spontaneous_regrow * scale
                num_neighbs = num_neighbs + self.spontaneous_regrow
                # num_neighbs=num_neighbs.at[350:356,98:102].set(1/40)

                num_neighbs = jnp.clip(num_neighbs - grid[:, :, 2], 0, 1)

                grid = grid.at[:, :, 1].add(random.bernoulli(next_key, num_neighbs))

            ####
            steps = state.steps + 1

            # decay of energy and clipping
            energy = energy - self.energy_decay + rewards
            energy = jnp.clip(energy, -1000, self.max_ener)

            time_good_level = jnp.where(energy > 0, (state.agents.time_good_level + 1) * alive, 0)

            time_alive = state.agents.time_alive

            # look if still alive

            time_alive = jnp.where(alive > 0, time_alive + 1, 0)

            # compute reproducer and go through the function only if there is one
            reproducer = jnp.where(state.agents.time_good_level > self.time_reproduce, 1, 0)
            next_key, key = random.split(key)

            nodes, conns, seqs, u_conns = state.agents.nodes, state.agents.conns, state.agents.seqs, state.agents.u_conns


            #TODO Fix this function
            nodes, conns, seqs, u_conns, posx, posy, energy, time_good_level, time_alive, alive, nb_food, nb_offspring = jax.lax.cond(
                reproducer.sum() > 0, mating_reproduce, lambda ai, px, py, n, c, s, u, e, tgl, k, ta, al, nf, no: (n, c, s, u, px, py, e, tgl, ta, al, nf, no),
                *(
                    action_int, posx, posy, nodes, conns, seqs, u_conns, energy, time_good_level, next_key,
                    time_alive, alive, nb_food, state.agents.nb_offspring))



            time_under_level = jnp.where(energy < 0, state.agents.time_under_level + 1, 0)
            alive = jnp.where(jnp.logical_or(time_alive > self.max_age, time_under_level > self.time_death), 0, alive)

            done = False
            steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)
            cur_state = State(state=grid, obs=get_obs_vector(grid, posx, posy), last_actions=actions,
                              rewards=jnp.expand_dims(rewards, -1),
                              agents=AgentStates(posx=posx, posy=posy, nodes=nodes, conns=conns, seqs=seqs, u_conns=u_conns,
                                                 energy=energy, time_good_level=time_good_level,
                                                 time_alive=time_alive, time_under_level=time_under_level, alive=alive,
                                                 nb_food=nb_food, nb_offspring=nb_offspring),
                              steps=steps, key=key)

            # keep it in case we let agent several trials
            state = jax.lax.cond(
                done, lambda x: reset_fn(state.key), lambda x: x, cur_state)

            return state, rewards, energy

        self._step_fn = jax.jit(step_fn)

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             ) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state)


