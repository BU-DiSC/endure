"""
This class defined the cost function of the LSM tree
"""
import numpy as np
from numba.experimental import jitclass
from numba import types

spec = [
    ('N', types.float64),
    ('phi', types.float64),
    ('s', types.float64),
    ('B', types.int64),
    ('E', types.int64),
    ('M', types.float64),
    ('is_leveling_policy', types.boolean),
    ('P', types.int64),
    ('z0', types.float64),
    ('z1', types.float64),
    ('q', types.float64),
    ('w', types.float64),
]

BITS_IN_BYTES = 8


@jitclass(spec)
class CostFunction:
    """
    This class defines the cost function of the LSM Tree
    """

    def __init__(self, N, phi, s, B, E, M, is_leveling_policy, z0, z1, q, w):
        self.N, self.phi, self.s, = N, phi, s
        self.B, self.E, self.M = B, E, M
        self.is_leveling_policy = is_leveling_policy
        self.z0, self.z1, self.q, self.w = z0, z1, q, w

    def L(self, h, T, get_ceiling=True):
        mbuff = self.M - (h * self.N)
        level = np.log(((self.N * self.E) / mbuff) + 1) / np.log(T)
        if get_ceiling:
            level = np.ceil(level)

        return level

    def N_full(self, L, h, T):
        num_entries = 0
        mbuff = self.M - (h * self.N)
        for level in range(1, int(L) + 1):
            num_entries += (T - 1) * (T ** (level - 1)) * mbuff / self.E

        return num_entries

    def fp(self, h, T, curr_level):
        alpha = (T ** (T / (T - 1))) / (T ** (self.L(h, T) + 1 - curr_level))

        return alpha * np.exp(-1 * h * (np.log(2) ** 2))

    def Z0(self, h, T):
        z0 = 0
        for i in range(1, self.L(h, T) + 1):
            z0 += self.fp(h, T, i)

        if not self.is_leveling_policy:
            z0 *= (T - 1)

        return z0

    def Z1(self, h, T):
        mbuff = self.M - (h * self.N)
        assert mbuff > 0, 'Mbuff must be positive'

        cost = 0
        L = self.L(h, T)
        Nf = self.N_full(L, h, T)
        def run_prob(i): return (mbuff * (T ** (i - 1))) / (Nf * self.E)

        if self.is_leveling_policy:
            for i in range(1, L + 1):
                fp_levels_sum = 0
                for k in range(1, i - 1):
                    fp_levels_sum += self.fp(h, T, k)
                cost += (T - 1) * run_prob(i) * (1 + fp_levels_sum)
        else:
            for i in range(1, L + 1):
                fp_levels_sum = 0
                for k in range(1, i - 1):
                    fp_levels_sum += self.fp(h, T, k)
                cost += (T - 1) * run_prob(i) * \
                    (1 + fp_levels_sum + ((T - 2) / 2) * self.fp(h, T, i))

        return cost

    def Q(self, h, T):
        q = self.s * self.N / self.B
        if self.is_leveling_policy:
            q += self.L(h, T, get_ceiling=False)
        else:
            q += (self.L(h, T, get_ceiling=False) * (T - 1))
        return q

    def W(self, h, T):
        w = (T - 1) * (1 + self.phi) * self.L(h, T, get_ceiling=False) / self.B
        if self.is_leveling_policy:
            w /= 2
        else:
            w /= T
        return w

    def calculate_cost(self, h, T, is_leveling_policy=None, B=None, E=None):
        if np.isnan(h):
            return np.iinfo(np.int64).max
        if np.isnan(T):
            return np.iinfo(np.int64).max

        if is_leveling_policy is not None:
            self.is_leveling_policy = is_leveling_policy
        if E is not None:
            self.E = E

        cost = ((self.z0 * self.Z0(h, T))
                + (self.z1 * self.Z1(h, T))
                + (self.q * self.Q(h, T))
                + (self.w * self.W(h, T)))

        return cost
