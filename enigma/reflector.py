from .transformation import Transformation

class Reflector(Transformation):
    def reflect(self, v):
        return self.apply(v)
