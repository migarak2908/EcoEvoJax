from reprod.adapted_neat.common import StatefulBaseClass, hash_array

import jax
import jax.numpy as jnp

class BaseGene(StatefulBaseClass):
    fixed_attrs = []
    custom_attrs = []

    def __init__(self):
        pass

    def new_identity_attrs(self, state):
        raise NotImplementedError

    def new_random_attrs(self, state, randkey):
        raise NotImplementedError

    def mutate(self, state, randkey, attr):
        raise NotImplementedError

    def crossover(self, state, randkey, attr1, attr2):
        return jnp.where(jax.random.normal(randkey, attr1.shape) > 0, attr1, attr2)

    def forward(self, state, attrs, inputs):
        raise NotImplementedError

    @property
    def length(self):
        return len(self.fixed_attrs) + len(self.custom_attrs)

    def repr(self, state, gene, precision=2):
        raise NotImplementedError

    def hash(self, gene):
        return hash_array(gene)