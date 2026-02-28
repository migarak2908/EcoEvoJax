from reprod.adapted_neat.common.state import State

class StatefulBaseClass:
    def setup(self, state=None):
        if state is None:
            state = State()
        return state

    def show_config(self, registered_objects=None):
        if registered_objects is None:
            registered_objects = []

        config = {}

        for key, value in self.__dict__.items():
            if isinstance(value, StatefulBaseClass) and value not in registered_objects:
                registered_objects.append(value)
                config[str(key)] = value.show_config(registered_objects)
            else:
                config[str(key)] = str(value)

        return config