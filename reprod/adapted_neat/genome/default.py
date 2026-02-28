import warnings

import jax
from jax import vmap, numpy as jnp
import numpy as np
import sympy as sp

from reprod.adapted_neat.genome.base import BaseGenome
from reprod.adapted_neat.genome.gene import DefaultNode, DefaultConn
from reprod.adapted_neat.genome.operations import DefaultMutation, DefaultCrossover
from reprod.adapted_neat.genome.utils import unflatten_conns, extract_gene_attrs

from reprod.adapted_neat.common import (
    topological_sort,
    topological_sort_python,
    find_useful_nodes,
    I_INF,
    attach_with_inf,
    ACT,
    AGG,
)


class DefaultGenome(BaseGenome):
    """Default genome class, with the same behavior as the NEAT-Python"""

    network_type = "feedforward"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes=50,
        max_conns=100,
        node_gene=DefaultNode(),
        conn_gene=DefaultConn(),
        mutation=DefaultMutation(),
        crossover=DefaultCrossover(),
        output_transform=None,
        input_transform=None,
        init_hidden_layers=()
    ):

        super().__init__(
            num_inputs,
            num_outputs,
            max_nodes,
            max_conns,
            node_gene,
            conn_gene,
            mutation,
            crossover,
            output_transform,
            input_transform,
            init_hidden_layers,
        )

    def transform(self, state, nodes, conns):
        u_conns = unflatten_conns(nodes, conns)
        conn_exist = u_conns != I_INF

        seqs = topological_sort(nodes, conn_exist)

        return seqs, nodes, conns, u_conns

    def network_dict(self, state, nodes, conns):
        network = super().network_dict(state, nodes, conns)
        topo_order, topo_layers = topological_sort_python(
            set(network["node"]), set(network["conns"])
        )
        network["topo_order"] = topo_order
        network["topo_layers"] = topo_layers
        network["useful_nodes"] = find_useful_nodes(
            set(network["node"]),
            set(network["conns"]),
            set(self.output_idx)
        )
        return network

    def forward(self, state, transformed, inputs):

        if self.input_transform is not None:
            inputs = self.input_transform(inputs)

        cal_seqs, nodes, conns, u_conns = transformed

        ini_vals = jnp.full((self.max_nodes,), jnp.nan)
        ini_vals = ini_vals.at[self.input_idx].set(inputs)
        nodes_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(self.node_gene, nodes)
        conns_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(self.conn_gene, conns)

        def cond_fun(carry):
            values, idx = carry
            return (idx < self.max_nodes) & (
                    cal_seqs[idx] != I_INF
            )  # not out of bounds and next node exists

        def body_func(carry):
            values, idx = carry
            i = cal_seqs[idx]

            def input_node():
                return values

            def otherwise():
                # calculate connections
                conn_indices = u_conns[:, i]
                hit_attrs = attach_with_inf(
                    conns_attrs, conn_indices
                )  # fetch conn attrs
                ins = vmap(self.conn_gene.forward, in_axes=(None, 0, 0))(
                    state, hit_attrs, values
                )

                # calculate node
                z = self.node_gene.forward(
                    state,
                    nodes_attrs[i],
                    ins,
                    is_output_node=jnp.isin(nodes[i, 0], self.output_idx)  # node[0] -> the key of node
                )

                # set new value
                new_values = values.at[i].set(z)
                return new_values

            values = jax.lax.cond(jnp.isin(i, self.input_idx), input_node, otherwise)

            return values, idx + 1

        vals, _ = jax.lax.while_loop(cond_fun, body_func, (ini_vals, 0))

        if self.output_transform is None:
            return vals[self.output_idx]
        else:
            return self.output_transform(vals[self.output_idx])