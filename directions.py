from ciabatta import vector


class Directions(object):

    def __init__(self, u_0):
        self.u = u_0
        self.n, self.dim = self.u.shape

    def rotate(self, ths):
        pass


class UniformDirections(Directions):

    def __init__(self, n, dim):
        u_0 = vector.sphere_pick(n=n, d=dim)
        Directions.__init__(self, u_0)
