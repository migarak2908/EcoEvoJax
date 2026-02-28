from reprod.adapted_neat.genome.gene import BaseGene


class BaseNode(BaseGene):
    fixed_attrs = ["idx"]

    def __init__(self):
        super().__init__()

    def forward(self, state, attrs, inputs):
        raise NotImplementedError

    def to_dict(self, state, node):
        idx = int(node[0])
        return {"idx": {idx}}

    def repr(self, state, node, precision=2, idx_width=3, func_width=8):
        idx = int(node[0])

        return "{}(idx: {:<{idx_width}})".format(
            self.__class__.__name__, idx, idx_width=idx_width
        )

    def sympy_func(self, state, node_dict, inputs, is_output_node=False):
        raise NotImplementedError

