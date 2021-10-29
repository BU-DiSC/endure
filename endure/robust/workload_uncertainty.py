"""
This class implements the robust linear program
"""

import logging
import numpy as np
# np.seterr(all='ignore')
from scipy.optimize import minimize, basinhopping, Bounds


class WorkloadUncertainty(object):
    """
    Robust non-linear program for workload uncertainty
    """
    def __init__(self, cf):
        """Constructor

        :param cf:
        """
        self.cf = cf
        self.logger = logging.getLogger("rlt_logger")
        self.rho = 0.

    def KL_divergence_conjugate(self, s):
        """Conjugate of divergence function

        :param s:
        """
        ret = np.exp(s) - 1

        # return min(1e6, ret)
        return ret

    def calculate_objective(self, x):
        """Calculates dual objective

        :param x:
        :return cost:
        """
        h = x[0]
        T = x[1]
        lamb = x[2]
        eta = x[3]

        total_cost = 0
        total_cost += self.cf.z0 * self.KL_divergence_conjugate((self.cf.Z0(h, T) - eta) / lamb)
        total_cost += self.cf.z1 * self.KL_divergence_conjugate((self.cf.Z1(h, T) - eta) / lamb)
        total_cost += self.cf.q * self.KL_divergence_conjugate((self.cf.Q(h, T) - eta) / lamb)
        total_cost += self.cf.w * self.KL_divergence_conjugate((self.cf.W(h, T) - eta) / lamb)
        cost = eta + (self.rho * lamb) + (lamb * total_cost)
        return cost

    def cf_callback(self, x):
        h, T, eta, lamb = x
        total = self.cf.calculate_cost(h, T)
        z0 = self.KL_divergence_conjugate((self.cf.Z0(h, T) - eta) / lamb)
        z1 = self.KL_divergence_conjugate((self.cf.Z1(h, T) - eta) / lamb)
        q = self.KL_divergence_conjugate((self.cf.Q(h, T) - eta) / lamb)
        w = self.KL_divergence_conjugate((self.cf.W(h, T) - eta) / lamb)

        print(f'{eta:.2f}\t {lamb:.2f}\t {z0:.2f}\t {z1:.2f}\t {q:.2f}\t {w:.2f}\t {self.calculate_objective(x):.6f}\t {total:.6f}\t {h:.6f}\t {T:.6f}')

    def get_robust_leveling_design(self, rho, workload=None, nominal_design=None):
        """Returns robust leveling design

        :param rho:
        :param workload:
        :param nominal_design:
        :return design:
        """
        self.rho = rho

        if workload is not None:
            self.cf.w = workload['w']
            self.cf.z0 = workload['z0']
            self.cf.z1 = workload['z1']
            self.cf.q = workload['q']

        one_mib_in_bits = 1024 * 1024 * 8

        if nominal_design is not None:
            h_initial = nominal_design['M_filt'] / self.cf.N
            T_initial = nominal_design['T']
        else:
            h_initial = 5.
            T_initial = 20.

        design = {}

        # Check leveling cost
        h_upper_lim = (self.cf.M / self.cf.N) - (one_mib_in_bits / self.cf.N)
        T_upper_lim = 100
        T_lower_lim = 2
        bounds = Bounds([1, T_lower_lim, 0.1, -np.inf], [h_upper_lim, T_upper_lim, np.inf, np.inf], keep_feasible=True)

        minimizer_kwargs = {
        'method' : 'SLSQP',
        'bounds' : bounds,
        'options': {'ftol': 1e-12, 'disp': False}}

        self.cf.is_leveling_policy = True
        sol = minimize(fun=self.calculate_objective,
                       x0=np.array([h_initial, T_initial, 1., 1.]),
                    #    callback = self.cf_callback,
                       **minimizer_kwargs)
        cost = self.cf.calculate_cost(sol.x[0], sol.x[1])
        design['exit_mode'] = sol.status
        design['T'] = sol.x[1]
        design['M_filt'] = sol.x[0] * self.cf.N
        design['M_buff'] = self.cf.M - design['M_filt']
        design['is_leveling_policy'] = True
        design['lambda'] = sol.x[2]
        design['eta'] = sol.x[3]
        design['cost'] = cost
        design['obj'] = sol.fun
        return design

    def get_robust_tiering_design(self, rho, workload=None, nominal_design=None):
        """Returns robust tiering design

        :param rho:
        :param workload:
        :param nominal_design:
        :return design:
        """
        self.rho = rho

        if workload is not None:
            self.cf.w = workload['w']
            self.cf.z0 = workload['z0']
            self.cf.z1 = workload['z1']
            self.cf.q = workload['q']

        one_mib_in_bits = 1024 * 1024 * 8

        if nominal_design is not None:
            h_initial = nominal_design['M_filt'] / self.cf.N
            T_initial = nominal_design['T']
        else:
            h_initial = 5.
            T_initial = 20.

        design = {}

        # Check tiering cost
        h_upper_lim = (self.cf.M / self.cf.N) - (one_mib_in_bits / self.cf.N)
        T_upper_lim = 100
        T_lower_lim = 2
        bounds = Bounds([1, T_lower_lim, 0.1, -np.inf], [h_upper_lim, T_upper_lim, np.inf, np.inf], keep_feasible=True)

        minimizer_kwargs = {
        'method' : 'SLSQP',
        'bounds' : bounds,
        'options': {'ftol': 1e-12, 'disp': False}}

        self.cf.is_leveling_policy = False
        sol = minimize(fun=self.calculate_objective,
                       x0=np.array([h_initial, T_initial, 1e20, 1]),
                    #    callback = self.cf_callback,
                       **minimizer_kwargs)

        cost = self.cf.calculate_cost(sol.x[0], sol.x[1])
        design['exit_mode'] = sol.status
        design['T'] = sol.x[1]
        design['M_filt'] = sol.x[0] * self.cf.N
        design['M_buff'] = self.cf.M - design['M_filt']
        design['is_leveling_policy'] = False
        design['lambda'] = sol.x[2]
        design['eta'] = sol.x[3]
        design['cost'] = cost
        design['obj'] = sol.fun
        return design
