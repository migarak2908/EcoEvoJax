from reprod.adapted_neat.common import StatefulBaseClass


class BaseMutation(StatefulBaseClass):
    def __call__(self, state, genome, randkey, nodes, conns, new_node_key, new_conn_key):
        raise NotImplementedError