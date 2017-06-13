class InvalidMeritFunction(Exception):
    pass


class MeritFunctionRegistry:
    registry = {}

    @classmethod
    def register(cls, merit):
        if not hasattr(merit, 'name'):
            raise InvalidMeritFunction("No name given")
        if merit.name in cls.registry:
            raise InvalidMeritFunction(f"Name {merit.name} is already registered to {cls.registry[merit.name]}")
        if not hasattr(merit, 'merit'):
            raise InvalidMeritFunction("No merit function specfied")
        if not hasattr(merit, 'weights'):
            merit.weights = property(lambda s: dict())
        if not hasattr(merit, 'model_params'):
            merit.model_params = property(lambda s: dict())
        if merit.__init__ is object.__init__:
            merit.__init__ = lambda s, ss: super(merit, s).__init__()
        if not hasattr(merit, 'configure'):
            merit.configure = lambda s, c: c
        if not hasattr(merit, 'configure_thread_model'):
            merit.configure_thread_model = lambda s, m, t: m
        cls.registry[merit.name] = merit
        return merit

    @classmethod
    def get(cls, item):
        return cls.registry[item]
