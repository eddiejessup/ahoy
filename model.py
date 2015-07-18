import numpy as np
from ciabatta.model import BaseModel
from ciabatta.fields import Space


class RectSpace(Space):

    def A(self):
        return np.product(self.L)


class Model(BaseModel):
    def __init__(self, seed, dt, agents):
        super(Model, self).__init__(seed, dt)
        self.dim = agents.dim
        self.agents = agents
        self.arena = RectSpace(agents.positions.L, self.dim)

        self.agent_density = self.agents.n / self.arena.A()

    def iterate(self):
        self.agents.iterate()
        super(Model, self).iterate()
