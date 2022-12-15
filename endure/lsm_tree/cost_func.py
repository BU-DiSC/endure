import numpy as np
import enum
from numba.experimental import jitclass
from numba import types

spec = [
    ('B', types.float64),
    ('E', types.float64),
    ('H', types.float64),
    ('N', types.float64),
    ('phi', types.float64),
    ('s', types.float64),
]

BITS_IN_BYTES = 8


class Policy(enum.Enum):
    Tiering = 0
    Leveling = 1
    KHybrid = 2
    QFixed = 3
    YZHybrid = 4


@jitclass(spec)
class EndureTierLevelCost:
    def __init__(
            self,
            B: float,
            E: float,
            H: float,
            N: float,
            phi: float,
            s: float
    ) -> None:
        self.B, self.E, self.H, self.N = B, E, H, N
        self.phi, self.s = phi, s

    def mbuff(self, h: float) -> float:
        return (((self.H - h) * self.N) / BITS_IN_BYTES)

    def L(self, h: float, T: float, ceil: bool = False) -> float:
        level = np.log(((self.N * self.E) / self.mbuff(h)) + 1) / np.log(T)
        return np.ceil(level) if ceil else level

    def fp(self, h: float, T: float, i: int) -> float:
        alpha = np.exp(-h * (np.log(2)**2))
        top = (T ** (T / (T - 1)))
        bot = (T**(self.L(h, T, ceil=True) + 1 - i))
        return alpha * (top / bot)

    def Nfull(self, h: float, T: float, levels: int) -> float:
        return sum([(T - 1) * (T ** (level - 1)) * self.mbuff(h) / self.E
                   for level in range(1, levels + 1)])

    def run_prob(self, level: int, T: float, mbuff: float, Nf: float) -> float:
        return (T - 1) * mbuff * T**(level - 1) / (Nf * self.E)

    def Z0(self, h: float, T: float, policy: Policy) -> float:
        z0 = sum([self.fp(h, T, level)
                 for level in range(1, int(self.L(h, T, ceil=True)) + 1)])
        if policy == Policy.Tiering:
            z0 *= (T - 1)
        return z0

    def Z1(self, h: float, T: float, policy: Policy) -> float:
        L = int(self.L(h, T, ceil=True))
        z1 = 0
        for i in range(1, L + 1):
            upper_fp = sum([self.fp(h, T, j) for j in range(1, i)])
            run_prob = self.run_prob(i, T, self.mbuff(h), self.Nfull(h, T, L))
            if policy == Policy.Tiering:
                upper_fp *= (T - 1)
                curr_fp = ((T - 2) / 2) * self.fp(h, T, i)
                z1 += run_prob * (1 + upper_fp + curr_fp)
            else:  # Policy.Leveling
                z1 += run_prob * (1 + upper_fp)

        return z1

    def Q(self, h: float, T: float, policy: Policy) -> float:
        q = self.s * self.N / self.B
        if policy == Policy.Tiering:
            q += (T - 1) * self.L(h, T, ceil=False)
        else:  # Policy.Leveling
            q += self.L(h, T, ceil=False)
        return q

    def W(self, h: float, T: float, policy: Policy) -> float:
        w = (1 + self.phi) * self.L(h, T, ceil=False) / self.B
        if policy == Policy.Leveling:
            w *= (T / 2)
        return w

    def calc_cost(
        self,
        h: float,
        T: float,
        policy: Policy,
        z0: float,
        z1: float,
        q: float,
        w: float
    ) -> float:
        if np.isnan(h) or np.isnan(T):
            return np.finfo(np.float64).max

        return ((z0 * self.Z0(h, T, policy))
                + (z1 * self.Z1(h, T, policy))
                + (q * self.Q(h, T, policy))
                + (w * self.W(h, T, policy)))


@jitclass(spec)
class EndureQFixedCost():
    def __init__(
        self,
        B: float,
        E: float,
        H: float,
        N: float,
        phi: float,
        s: float
    ) -> None:
        self.B, self.E, self.H, self.N = B, E, H, N
        self.phi, self.s = phi, s

    def mbuff(self, h: float) -> float:
        return (((self.H - h) * self.N) / BITS_IN_BYTES)

    def L(self, h: float, T: float, ceil: bool = False) -> float:
        level = np.log(((self.N * self.E) / self.mbuff(h)) + 1) / np.log(T)
        return np.ceil(level) if ceil else level

    def fp(self, h: float, T: float, i: int) -> float:
        alpha = np.exp(-h * (np.log(2)**2))
        top = (T ** (T / (T - 1)))
        bot = (T**(self.L(h, T, ceil=True) + 1 - i))
        return alpha * (top / bot)

    def Nfull(self, h: float, T: float, levels: int) -> float:
        return sum([(T - 1) * (T ** (level - 1)) * self.mbuff(h) / self.E
                   for level in range(1, levels + 1)])

    def run_prob(self, level: int, T: float, mbuff: float, Nf: float) -> float:
        return (T - 1) * mbuff * T**(level - 1) / (Nf * self.E)

    def Z0(self, h: float, T: float, Q: float) -> float:
        z0 = 0
        for level in range(1, int(self.L(h, T, ceil=True)) + 1):
            z0 += Q * self.fp(h, T, level)

        return z0

    def Z1(self, h: float, T: float, Q: float) -> float:
        L = int(self.L(h, T, ceil=True))
        mbuff = self.mbuff(h)
        Nf = self.Nfull(h, T, L)

        z1 = 0
        for level in range(1, L + 1):
            upper_fp = 0
            for j in range(1, level):
                upper_fp += Q * self.fp(h, T, j)
            current_fp = ((Q - 1) / 2) * self.fp(h, T, level)
            z1 += (self.run_prob(level, T, mbuff, Nf)
                   * (1 + upper_fp + current_fp))

        return z1

    def Q(self, h: float, T: float, Q: float) -> float:
        return (Q * self.L(h, T)) + (self.s * self.N / self.B)

    def W(self, h: float, T: float, Q: float) -> float:
        return self.L(h, T) * (T - 1 + Q) * (1 + self.phi) / (2 * Q * self.B)

    def calc_cost(
        self,
        h: float,
        T: float,
        Q: float,
        z0: float,
        z1: float,
        q: float,
        w: float
    ) -> float:
        if np.isnan(h) or np.isnan(T) or np.isnan(Q):
            return np.finfo(np.float64).max

        cost = ((z0 * self.Z0(h, T, Q))
                + (z1 * self.Z1(h, T, Q))
                + (q * self.Q(h, T, Q))
                + (w * self.W(h, T, Q)))

        return cost


@jitclass(spec)
class EndureKHybridCost():
    def __init__(
        self,
        B: float,
        E: float,
        H: float,
        N: float,
        phi: float,
        s: float
    ) -> None:
        self.B, self.E, self.H, self.N = B, E, H, N
        self.phi, self.s = phi, s

    def mbuff(self, h: float) -> float:
        return (((self.H - h) * self.N) / BITS_IN_BYTES)

    def L(self, h: float, T: float, ceil: bool = False) -> float:
        level = np.log(((self.N * self.E) / self.mbuff(h)) + 1) / np.log(T)
        return np.ceil(level) if ceil else level

    def fp(self, h: float, T: float, i: int) -> float:
        alpha = np.exp(-h * (np.log(2)**2))
        top = (T ** (T / (T - 1)))
        bot = (T**(self.L(h, T, ceil=True) + 1 - i))
        return alpha * (top / bot)

    def Nfull(self, h: float, T: float, levels: int) -> float:
        return sum([(T - 1) * (T ** (level - 1)) * self.mbuff(h) / self.E
                   for level in range(1, levels + 1)])

    def run_prob(self, level: int, T: float, mbuff: float, Nf: float) -> float:
        return (T - 1) * mbuff * T**(level - 1) / (Nf * self.E)

    def Z0(self, h: float, T: float, K: list[float]) -> float:
        z0 = 0
        for i in range(1, int(self.L(h, T, ceil=True)) + 1):
            z0 += K[i - 1] * self.fp(h, T, i)

        return z0

    def Z1(self, h: float, T: float, K: list[float]) -> float:
        L = int(self.L(h, T, ceil=True))
        mbuff = self.mbuff(h)
        nfull = self.Nfull(h, T, L)

        z1 = 0
        for level in range(1, L + 1):
            upper_fp = 0
            run_prob = self.run_prob(level, T, mbuff, nfull)
            level_fp = self.fp(h, T, level)
            for j in range(1, level):
                upper_fp += K[j - 1] * self.fp(h, T, j)
            current_fp = ((K[level - 1] - 1) / 2) * level_fp
            z1 += run_prob * (1 + upper_fp + current_fp)

        return z1

    def Q(self, h: float, T: float, K: list[float]) -> float:
        L = int(self.L(h, T, ceil=True))
        return (self.s * self.N / self.B) + sum(K[:L])

    def W(self, h: float, T: float, K: list[float]) -> float:
        L = int(self.L(h, T, ceil=True))
        w = 0
        for level in range(0, L):
            w += (T - 1 + K[level]) / (2 * K[level])
        w *= (1 + self.phi) / self.B
        return w

    def calc_cost(
        self,
        h: float,
        T: float,
        K,  # List of values corresponding to file per level
        z0: float,
        z1: float,
        q: float,
        w: float
    ) -> float:
        if np.isnan(h) or np.isnan(T):
            return np.finfo(np.float64).max

        cost = ((z0 * self.Z0(h, T, K))
                + (z1 * self.Z1(h, T, K))
                + (q * self.Q(h, T, K))
                + (w * self.W(h, T, K)))

        return cost


@jitclass(spec)
class EndureYZHybridCost():
    def __init__(
        self,
        B: float,
        E: float,
        H: float,
        N: float,
        phi: float,
        s: float
    ) -> None:
        self.B, self.E, self.H, self.N = B, E, H, N
        self.phi, self.s = phi, s

    def mbuff(self, h: float) -> float:
        return (((self.H - h) * self.N) / BITS_IN_BYTES)

    def L(self, h: float, T: float, ceil: bool = False) -> float:
        level = np.log(((self.N * self.E) / self.mbuff(h)) + 1) / np.log(T)
        return np.ceil(level) if ceil else level

    def fp(self, h: float, T: float, i: int) -> float:
        alpha = np.exp(-h * (np.log(2)**2))
        top = (T ** (T / (T - 1)))
        bot = (T**(self.L(h, T, ceil=True) + 1 - i))
        return alpha * (top / bot)

    def Nfull(self, h: float, T: float, levels: int) -> float:
        return sum([(T - 1) * (T ** (level - 1)) * self.mbuff(h) / self.E
                   for level in range(1, levels + 1)])

    def run_prob(self, level: int, T: float, mbuff: float, Nf: float) -> float:
        return (T - 1) * mbuff * T**(level - 1) / (Nf * self.E)

    def Z0(self, h: float, T: float, Y: float, Z: float) -> float:
        z0 = 0
        L = int(self.L(h, T, ceil=True))
        for level in range(1, L):
            z0 += Y * self.fp(h, T, level)
        z0 += Z * self.fp(h, T, L)

        return z0

    def Z1(self, h: float, T: float, Y: float, Z: float) -> float:
        L = int(self.L(h, T, ceil=True))
        mbuff = self.mbuff(h)
        Nf = self.Nfull(h, T, L)

        z1 = 0
        for level in range(1, L):
            upper_fp = 0
            for j in range(1, level):
                upper_fp += Y * self.fp(h, T, j)
            current_fp = ((Y - 1) / 2) * self.fp(h, T, level)
            z1 += self.run_prob(level, T, mbuff, Nf) * \
                (1 + upper_fp + current_fp)

        upper_fp = 0
        for j in range(1, L):
            upper_fp += Y * self.fp(h, T, j)
        current_fp = ((Z - 1) / 2) * self.fp(h, T, L)
        z1 += self.run_prob(L, T, mbuff, Nf) * (1 + upper_fp + current_fp)

        return z1

    def Q(self, h: float, T: float, Y: float, Z: float) -> float:
        q = self.s * self.N / self.B
        q += Y * self.L(h, T, ceil=True) - 1
        q += Z

        return q

    def W(self, h: float, T: float, Y: float, Z: float) -> float:
        levels = self.L(h, T, ceil=True)
        w = (levels - 1) * (T - 1 + Y) / (2 * Y)  # middle levels
        w += (T - 1 + Z) / (2 * Z)  # last level is different
        w *= (1 + self.phi) / self.B

        return w

    def calc_cost(
        self,
        h: float,
        T: float,
        Y: float,
        Z: float,
        z0: float,
        z1: float,
        q: float,
        w: float
    ) -> float:
        if np.isnan(h) or np.isnan(T) or np.isnan(Y) or np.isnan(Z):
            return np.finfo(np.float64).max

        cost = ((z0 * self.Z0(h, T, Y, Z))
                + (z1 * self.Z1(h, T, Y, Z))
                + (q * self.Q(h, T, Y, Z))
                + (w * self.W(h, T, Y, Z)))

        return cost
