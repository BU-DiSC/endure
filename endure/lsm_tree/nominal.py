"""
This class implements the nominal optimization problem
"""

import logging
import numpy as np
np.seterr(all='ignore')
from scipy.optimize import minimize, basinhopping

class NominalWorkloadTuning(object):
    """
    Nominal non-linear program for workload uncertainty
    """

    def __init__(self, cost_func) -> None:
        self.cost_func = cost_func
        self.logger = logging.getLogger('rlt_logger')

    def calculate_objective(self, args):
        h, T = args

        total_cost = self.cost_func.calculate_cost(h, T)

        return total_cost

    def cf_callback(self, x):
        h, T, = x
        total = self.cost_func.calculate_cost(h, T)

        print(f'{total:.6f}\t {h:.6f}\t {T:.6f}')

    def get_nominal_design(self, is_leveling_policy=None, workload=None):
        T_UPPER_LIM, T_LOWER_LIM = (100, 2)
        one_mib_in_bits = 1024 * 1024 * 8
        H_UPPER_LIM = (self.cost_func.M / self.cost_func.N) - (one_mib_in_bits / self.cost_func.N)

        if workload is not None:
            self.cost_func.w = workload['w']
            self.cost_func.z0 = workload['z0']
            self.cost_func.z1 = workload['z1']
            self.cost_func.q = workload['q']

        h_initial = 5
        T_initial = 20.

        bounds = ((0, H_UPPER_LIM), (T_LOWER_LIM, T_UPPER_LIM))
        min_cost = np.inf
        design = {}
        minimizer_kwargs = {
            'method' : 'SLSQP',
            'bounds' : bounds,
            'options': {'ftol': 1e-6, 'disp': False}}

        # Check leveling cost
        if (is_leveling_policy is None) or (is_leveling_policy is True):
            self.cost_func.is_leveling_policy = True
            sol = minimize(fun=self.calculate_objective,
                           x0=np.array([h_initial, T_initial]),
                        #    callback=self.cf_callback,
                           **minimizer_kwargs)
            cost = self.cost_func.calculate_cost(sol.x[0], sol.x[1])
            if cost < min_cost:
                design['T'] = sol.x[1]
                design['M_filt'] = sol.x[0] * self.cost_func.N
                design['M_buff'] = self.cost_func.M - design['M_filt']
                design['is_leveling_policy'] = True
                design['cost'] = cost
                min_cost = cost

        # Check tiering cost
        if (is_leveling_policy is None) or (is_leveling_policy is False):
            self.cost_func.is_leveling_policy = False
            sol = minimize(fun=self.calculate_objective,
                           x0=np.array([h_initial, T_initial]),
                        #    callback=self.cf_callback,
                           **minimizer_kwargs)
            cost = self.cost_func.calculate_cost(sol.x[0], sol.x[1])
            if cost < min_cost:
                design['T'] = sol.x[1]
                design['M_filt'] = sol.x[0] * self.cost_func.N
                design['M_buff'] = self.cost_func.M - design['M_filt']
                design['is_leveling_policy'] = False
                design['cost'] = cost
                min_cost = cost

        return design
