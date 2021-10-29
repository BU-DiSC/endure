"""
Create workload uncertainty tunings
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import chi2
from copy import deepcopy
from lsm_tree.cost_function import CostFunction
from lsm_tree.nominal import NominalWorkloadTuning
from data.data_exporter import DataExporter


class CreateNominalWorkloadTunings(object):
    """
    Computes and exports tunings for nominal workload
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
        self.data_exporter = DataExporter(self.config)

    def run(self, alphas_list=None):
        """
        Runs the job
        """
        self.logger.info("Starting job: Create Workload Uncertainty Tunings")

        expected_workloads = self.config['expected_workloads']
        expected_memory_bits_per_element = self.config['expected_memory_bits_per_element']

        # Create a dataframe to store results
        df = []

        for idx in enumerate(expected_workloads):
            tmp = {}
            tmp['workload_idx'] = idx[0]
            w = idx[1]
            self.logger.info("Workload: {}".format(w))
            tmp['z0'] = w['z0']
            tmp['z1'] = w['z1']
            tmp['q'] = w['q']
            tmp['w'] = w['w']

            for m in expected_memory_bits_per_element:
                self.logger.info(f'Expected Bits per Element : {m}')
                self.config['lsm_tree_config']['M'] = m * self.config['lsm_tree_config']['N']
                tmp['N'] = self.config['lsm_tree_config']['N']
                tmp['phi'] = self.config['lsm_tree_config']['phi']
                tmp['B'] = self.config['lsm_tree_config']['B']
                tmp['s'] = self.config['lsm_tree_config']['s']
                tmp['E'] = self.config['lsm_tree_config']['E']
                tmp['M'] = self.config['lsm_tree_config']['M']

                cf = CostFunction(**self.config['lsm_tree_config'], **w)
                nominal = NominalWorkloadTuning(cf)
                nominal_design = nominal.get_nominal_design(is_leveling_policy=None)
                tmp['nominal_m_filt'] = nominal_design['M_filt']
                tmp['nominal_m_buff'] = nominal_design['M_buff']
                tmp['nominal_T'] = nominal_design['T']
                tmp['nominal_cost'] = nominal_design['cost']
                tmp['nominal_is_leveling_policy'] = nominal_design['is_leveling_policy']

                df.append(deepcopy(tmp))

        df = pd.DataFrame(df)
        self.data_exporter.export_csv_file(df, 'workload_nominal_tunings.csv')

        self.logger.info("Finished job: Create Workload Nominal Tunings\n")
        return df
