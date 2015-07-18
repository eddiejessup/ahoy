import numpy as np
from ciabatta import vector


class Rudders(object):

    def __init__(self, dt):
        self.dt = dt

    def rotate(self, directions, noise):
        return directions


class TumbleRudders(Rudders):

    def rotate(self, directions, noise):
        tumbles = np.random.uniform(size=directions.n) < noise * self.dt
        directions.u[tumbles] = vector.sphere_pick(directions.dim,
                                                   tumbles.sum())
        return directions


class RotationRudders(Rudders):

    def rotate(self, directions, noise):
        return directions
