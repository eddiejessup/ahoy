class RudderControllers(object):

    def __init__(self, rudders, noise_0):
        self.rudders = rudders
        self.noise_0 = noise_0

    def rotate(self, positions, directions):
        return self.rudders.rotate(directions, self.noise_0)


class ChemoRudderControllers(RudderControllers):

    def __init__(self, rudders, noise_0, chi, onesided_flag, estimators):
        RudderControllers.__init__(self, rudders, noise_0)
        self.chi = chi
        self.onesided_flag = onesided_flag
        self.estimators = estimators

    def rotate(self, positions, directions):
        alignments = self.estimators.get_alignments(positions, directions)
        self.rudders.rotate(directions, noise)
