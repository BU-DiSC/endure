"""
Create workload uncertainty tunings
"""

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from lsm_tree.cost_function import CostFunction
from lsm_tree.nominal import NominalWorkloadTuning
from robust.workload_uncertainty import WorkloadUncertainty
from data.data_exporter import DataExporter

class CreateWorkloadUncertaintyTunings(object):
    """
    Computes and exports tunings for workload uncertainty
    """
    def __init__(self, config):
        """
        Constructor

        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
        self.data_exporter = DataExporter(self.config)

    def create_rho_list(self) -> list:
        stop = self.config['uncertain_workload_config']['rho_high']
        start = self.config['uncertain_workload_config']['rho_low']
        step = self.config['uncertain_workload_config']['rho_step']
        rhos = np.arange(start, stop, step)

        return rhos

    def run(self):
        """
        Runs the job
        """
        self.logger.info("Starting job: Create Workload Uncertainty Tunings Varying Rho")

        expected_workloads = self.config['expected_workloads']
        expected_memory_bits_per_element = self.config['expected_memory_bits_per_element']
        rhos = self.create_rho_list()

        # Create a dataframe to store results
        df = []
        expected_workloads_pbar = tqdm(expected_workloads, desc='WL', ncols=120)

        for idx, w in enumerate(expected_workloads_pbar):
            expected_workloads_pbar.set_postfix(workload=w)
            row = {}
            row['workload_idx'] = idx
            row['z0'] = w['z0']
            row['z1'] = w['z1']
            row['q'] = w['q']
            row['w'] = w['w']

            bpe_pbar = tqdm(expected_memory_bits_per_element, desc='BPE', ncols=120, leave=False)
            for m in bpe_pbar:
                bpe_pbar.set_postfix(bits_per_element=m)
                self.config['lsm_tree_config']['M'] = m * self.config['lsm_tree_config']['N']
                row['N'] = self.config['lsm_tree_config']['N']
                row['phi'] = self.config['lsm_tree_config']['phi']
                row['B'] = self.config['lsm_tree_config']['B']
                row['s'] = self.config['lsm_tree_config']['s']
                row['E'] = self.config['lsm_tree_config']['E']
                row['M'] = self.config['lsm_tree_config']['M']

                cf = CostFunction(**self.config['lsm_tree_config'], **w)

                nominal = NominalWorkloadTuning(cf)
                nominal_design = nominal.get_nominal_design(is_leveling_policy=None)
                row['nominal_m_filt'] = nominal_design['M_filt']
                row['nominal_m_buff'] = nominal_design['M_buff']
                row['nominal_T'] = nominal_design['T']
                row['nominal_cost'] = nominal_design['cost']
                row['nominal_is_leveling_policy'] = nominal_design['is_leveling_policy']

                robust = WorkloadUncertainty(cf)
                for rho in rhos:
                    row['rho'] = rho
                    tiering_design = robust.get_robust_tiering_design(rho, nominal_design=None)
                    leveling_design = robust.get_robust_leveling_design(rho, nominal_design=None)
                    if tiering_design['obj'] < leveling_design['obj']:
                        robust_design = tiering_design
                    else:
                        robust_design = leveling_design
                    row['robust_exit_mode'] = robust_design['exit_mode']
                    row['robust_m_filt'] = robust_design['M_filt']
                    row['robust_m_buff'] = robust_design['M_buff']
                    row['robust_T'] = robust_design['T']
                    row['robust_cost'] = robust_design['cost']
                    row['robust_is_leveling_policy'] = robust_design['is_leveling_policy']

                    # Append the design to the dataframe
                    df.append(deepcopy(row))

        df = pd.DataFrame(df)
        self.data_exporter.export_csv_file(df, 'workload_uncertainty_tunings.csv')

        self.logger.info("Finished job: Create Workload Uncertainty Tunings\n")
        return df
