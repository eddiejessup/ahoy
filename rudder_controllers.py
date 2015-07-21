from __future__ import print_function, division
import numpy as np
import ships


class RudderControllers(object):

    def __init__(self, rudders, noise_0):
        self.rudders = rudders
        self.noise_0 = noise_0

    def get_noise(self, positions, directions):
        return self.noise_0

    def rotate(self, positions, directions):
        noise = self.get_noise(positions, directions)
        return self.rudders.rotate(directions, noise)

    def __repr__(self):
        return 'RC_{}_n0={:g}'.format(self.rudders, self.noise_0)


class ChemoRudderControllers(RudderControllers):

    def __init__(self, rudders, noise_0, v_0, chi, estimators):
        RudderControllers.__init__(self, rudders, noise_0)
        self.v_0 = v_0
        self.chi = chi
        self.estimators = estimators

    def get_noise(self, positions, directions):
        cdots = self.estimators.get_cdots(positions, directions)
        fitness = self.chi * cdots / self.v_0
        return self.noise_0 * (1.0 - fitness)

    def __repr__(self):
        repr_str = 'CRC_2side_{}_n0={:g}_chi={:g}_est={}'
        return repr_str.format(self.rudders, self.noise_0, self.chi,
                               self.estimators)


class OneSidedChemoRudderControllers(ChemoRudderControllers):

    def get_noise(self, positions, directions):
        noise_two_sided = super(OneSidedChemoRudderControllers,
                                self).get_noise(positions, directions)
        return np.minimum(self.noise_0, noise_two_sided)

    def __repr__(self):
        repr_str = 'CRC_1side_{}_n0={:g}_chi={:g}_est={}'
        return repr_str.format(self.rudders, self.noise_0, self.chi,
                               self.estimators)


def chemo_rud_conts_factory(onesided_flag, *args, **kwargs):
    if onesided_flag:
        return ChemoRudderControllers(*args, **kwargs)
    else:
        return OneSidedChemoRudderControllers(*args, **kwargs)


def rud_conts_factory(chemo_flag, onesided_flag, ruds, noise_0, v_0, chi,
                      esters):
    if chemo_flag:
        rud_conts = chemo_rud_conts_factory(onesided_flag, ruds, noise_0, v_0,
                                            chi, esters)
    else:
        rud_conts = RudderControllers(ruds, noise_0)
    return rud_conts


def rud_cont_sets_factory(dim, dt, v_0, p_0, chi, onesided_flag,
                          tumble_chemo_flag, D_rot_0, D_rot_chemo_flag, rng):
    # If no base source of noise to modulate, then no way to do chemotaxis.
    if not p_0:
        tumble_chemo_flag = False
    if not D_rot_0:
        D_rot_chemo_flag = False
    # If chi is zero, no point doing chemotaxis
    if not chi:
        tumble_chemo_flag = D_rot_chemo_flag = False
    chemo_flag = tumble_chemo_flag or D_rot_chemo_flag

    # If chemotaxis is happening, will need estimators.
    if chemo_flag:
        esters = ships.estimators.LinearSpatialCDotEstimators(v_0)

    rudder_controller_sets = []
    if p_0:
        tumble_ruds = ships.rudders.TumbleRudders(dt, rng)
        tumble_rud_conts = rud_conts_factory(tumble_chemo_flag, onesided_flag,
                                             tumble_ruds, p_0, v_0, chi,
                                             esters)
        rudder_controller_sets.append(tumble_rud_conts)
    if D_rot_0:
        rotation_ruds = ships.rudders.rotation_rudders_factory(dt, dim, rng)
        rotation_rud_conts = rud_conts_factory(D_rot_chemo_flag, onesided_flag,
                                               rotation_ruds, D_rot_0, v_0,
                                               chi, esters)
        rudder_controller_sets.append(rotation_rud_conts)
    return rudder_controller_sets
