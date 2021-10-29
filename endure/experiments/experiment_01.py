"""
Experiment 01
Compares nominal and robust performance for uncertainty using various values
of rho, workloads and memory_bits_per_element

for workload uncertainty comparisons
"""

import logging
from copy import deepcopy
import warnings
# warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
# np.seterr(all='ignore')
import pandas as pd
from tqdm import tqdm
from scipy.special import rel_entr

from data.data_provider import DataProvider
from data.data_exporter import DataExporter
from jobs.create_workload_uncertainty_tunings import CreateWorkloadUncertaintyTunings
from jobs.sample_uncertain_workloads import SampleUncertainWorkloads
from lsm_tree.cost_function import CostFunction


class Experiment01(object):
    """
    Experiment 01 class
    """

    def __init__(self, config):
        """
        Constructor

        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
        self.data_provider = DataProvider(self.config)
        self.data_exporter = DataExporter(self.config)
        self.expt_config = self.config['experiment_01']

    def run(self):
        """
        Run experiment
        """
        self.logger.info("Starting Experiment 01\n")
        ops_mask = (True, True, True, True)

        # Expected workloads list
        expected_workloads = [
            {'z0': 0.25, 'z1': 0.25, 'q': 0.25, 'w': 0.25},     # 00
            {'z0': 0.97, 'z1': 0.01, 'q': 0.01, 'w': 0.01},     # 01
            {'z0': 0.01, 'z1': 0.97, 'q': 0.01, 'w': 0.01},     # 02
            {'z0': 0.01, 'z1': 0.01, 'q': 0.97, 'w': 0.01},     # 03
            {'z0': 0.01, 'z1': 0.01, 'q': 0.01, 'w': 0.97},     # 04
            {'z0': 0.49, 'z1': 0.49, 'q': 0.01, 'w': 0.01},     # 01
            {'z0': 0.49, 'z1': 0.01, 'q': 0.49, 'w': 0.01},     # 06
            {'z0': 0.49, 'z1': 0.01, 'q': 0.01, 'w': 0.49},     # 07
            {'z0': 0.01, 'z1': 0.49, 'q': 0.49, 'w': 0.01},     # 08
            {'z0': 0.01, 'z1': 0.49, 'q': 0.01, 'w': 0.49},     # 09
            {'z0': 0.01, 'z1': 0.01, 'q': 0.49, 'w': 0.49},     # 10
            {'z0': 0.33, 'z1': 0.33, 'q': 0.33, 'w': 0.01},     # 11
            {'z0': 0.33, 'z1': 0.33, 'q': 0.01, 'w': 0.33},     # 12
            {'z0': 0.33, 'z1': 0.01, 'q': 0.33, 'w': 0.33},     # 13
            {'z0': 0.01, 'z1': 0.33, 'q': 0.33, 'w': 0.33},     # 14
            {'z0': 0.10, 'z1': 0.10, 'q': 0.10, 'w': 0.70},     # 15
            {'z0': 0.70, 'z1': 0.20, 'q': 0.01, 'w': 0.01},     # 16
            {'z0': 0.30, 'z1': 0.01, 'q': 0.01, 'w': 0.60},     # 17
       ]

        # Expected memory bits per element list
        # expected_memory_bits_per_element = [10, 15, 20]
        expected_memory_bits_per_element = [10]

        # Sample size
        sample_size = 10000

        # Create workload uncertainty tunings
        config = deepcopy(self.config)
        config['expected_workloads'] = expected_workloads
        config['expected_memory_bits_per_element'] = expected_memory_bits_per_element
        config['uncertain_workload_config']['rho_low'] = 0
        config['uncertain_workload_config']['rho_high'] = 4
        config['uncertain_workload_config']['rho_step'] = 0.25
        config['uncertain_workload_config']['N'] = sample_size
        config['lsm_tree_config']['N'] = 1e8

        # Sample uncertain workloads object
        suw = SampleUncertainWorkloads(config)

        # Create workload uncertainty tunings and get a list of tunings dictionaries
        cwut = CreateWorkloadUncertaintyTunings(config)
        tunings = cwut.run().to_dict('records')

        comparisons = []
        sample_wls = suw.get_uncertain_samples(sample_size, ops_mask)

        # Calculating distances for all expected WLs
        self.logger.info('Calcuting rho hat values for all expected workloads')
        distances = {}
        for wl in expected_workloads:
            key = str(wl)
            w0 = [wl['z0'], wl['z1'], wl['q'], wl['w']]
            w0_tmp = [op for op, mask in list(zip(w0, ops_mask)) if mask]
            distances[key] = []
            for sample in sample_wls:
                w_hat = [sample['z0'], sample['z1'], sample['q'], sample['w']]
                w_hat_tmp = [op for op, mask in list(zip(w_hat, ops_mask)) if mask]
                distances[key].append(np.sum(rel_entr(w_hat_tmp, w0_tmp)))

        self.logger.info('Calculating cost of tunings')
        for tuning in tqdm(tunings, desc='Tunings', ncols=120):
            row = {}
            row['workload_idx'] = tuning['workload_idx']
            row['w'] = {'z0': tuning['z0'], 'z1': tuning['z1'], 'q': tuning['q'], 'w': tuning['w']}
            row['N'] = tuning['N']
            row['M'] = tuning['M']

            # Tunings
            row['robust_rho'] = tuning['rho']
            row['robust_m_filt'] = tuning['robust_m_filt']
            row['robust_T'] = tuning['robust_T']
            row['robust_is_leveling_policy'] = tuning['robust_is_leveling_policy']
            row['robust_exit_mode'] = tuning['robust_exit_mode']

            row['nominal_m_filt'] = tuning['nominal_m_filt']
            row['nominal_T'] = tuning['nominal_T']
            row['nominal_is_leveling_policy'] = tuning['nominal_is_leveling_policy']

            config['lsm_tree_config']['M'] = tuning['M']

            distance = distances[str(row['w'])]
            for idx, w_hat in enumerate(tqdm(sample_wls, desc='Sample Workloads', ncols=120, leave=False)):
                row['rho_hat'] = distance[idx]
                row['w_hat'] = w_hat
                row['sample_idx'] = idx

                # Get nominal cost
                config['lsm_tree_config']['is_leveling_policy'] = row['nominal_is_leveling_policy']
                cf = CostFunction(**config['lsm_tree_config'], **w_hat)
                nominal_cost = cf.calculate_cost(row['nominal_m_filt'] / row['N'], row['nominal_T'])
                row['nominal_cost'] = nominal_cost
                del cf

                # Get robust cost
                config['lsm_tree_config']['is_leveling_policy'] = row['robust_is_leveling_policy']
                cf = CostFunction(**config['lsm_tree_config'], **w_hat)
                robust_cost = cf.calculate_cost(row['robust_m_filt'] / row['N'], row['robust_T'])
                row['robust_cost'] = robust_cost
                del cf

                comparisons.append(deepcopy(row))

        df = pd.DataFrame(comparisons)
        self.logger.info("Exporting data from experiment 01")
        self.data_exporter.export_csv_file(df, 'experiment_01.csv')
        self.logger.info("Finished Experiment 01\n")
