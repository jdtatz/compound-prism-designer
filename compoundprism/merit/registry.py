class InvalidMeritFunction(Exception):
    pass


class MeritFunctionRegistry(type):
    registry = {}

    def __new__(cls, name, bases, namespace):
        if not len(bases):
            return super().__new__(cls, name, bases, namespace)
        if 'name' not in namespace:
            raise InvalidMeritFunction("No name given")
        reg_name = namespace["name"]
        if reg_name in cls.registry:
            raise InvalidMeritFunction(f"Name {reg_name} is already registered to {cls.registry[reg_name]}")
        if 'merit' not in namespace:
            raise InvalidMeritFunction("No merit function specfied")
        if not isinstance(namespace['merit'], staticmethod):
            raise InvalidMeritFunction("The merit function must be a static method")
        new_cls = super().__new__(cls, name, bases, namespace)
        cls.registry[reg_name] = new_cls
        return new_cls

    @classmethod
    def get(mcs, item):
        return mcs.registry[item]


class BaseMeritFunction(metaclass=MeritFunctionRegistry):
    weights = {}
    model_params = {}

    def __init__(self, settings):
        pass

    def configure(self, configuration):
        return configuration

    def configure_thread_model(self, model, tid):
        return model
