from reprod.adapted_neat.genome.gene import BaseGene


class BaseConn(BaseGene):
    fixed_attrs = ["input_idx", "output_idx"]

    def __init__(self):
        super().__init__()

    def new_zero_attrs(self, state):
        raise NotImplementedError

    def forward(self, state, attrs, inputs):
        raise NotImplementedError

    def repr(self, state, conn, precision=2, idx_width=3, func_width=8):
        in_idx, out_idx = conn[:2]
        in_idx = int(in_idx)
        out_idx = int(out_idx)

        return "{}(in: {:<{idx_width}}, out: {:<{idx_width}})".format(
            self.__class__.__name__, in_idx, out_idx, idx_width=idx_width
        )

    def to_dict(self, state, conn):
        in_idx, out_idx = conn[:2]
        return {
            "in": int(in_idx),
            "out": int(out_idx),
        }



