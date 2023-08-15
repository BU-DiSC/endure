"""
This class implements the default optimization problem
"""

from scipy.optimize import minimize
import logging
import numpy as np
np.seterr(all='ignore')


class DefaultWorkloadTuning(object):
    """
    Default non-linear program for workload uncertainty
    """

    def __init__(self, cost_func) -> None:
        self.cost_func = cost_func
        self.logger = logging.getLogger('rlt_logger')


    def get_default_design(self):
        cost = self.cost_func.calculate_cost(10, 10)
        design ={}
        design['T'] = 10
        design['M_h'] = 10
        design['M_filt'] = 100000000
        design['M_buff'] = 8589934592
        design['is_leveling_policy'] = True
        design['filter_policy'] = 0
        design['E'] = 8192
        design['cost'] = cost

        return design

    def get_super_default_design(self):
        cost = self.cost_func.calculate_cost(10, 10)
        design ={}
        design['T'] = 10
        design['M_h'] = 10
        design['M_filt'] = 100000000
        design['M_buff'] = 8589934592
        design['is_leveling_policy'] = True
        design['filter_policy'] = 1
        design['E'] = 8192
        design['cost'] = cost

        return design



