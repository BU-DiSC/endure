"""
Samples workloads from an uncertainity bubble
"""

import logging
import numpy as np
from data.data_exporter import DataExporter
from scipy.special import rel_entr

PRECISION = 4


class SampleUncertainWorkloads(object):
    """
    Samples workloads
    """
    def __init__(self, config):
        """Constructor

        :param config: Holds all configurations for all experiments
        :type config: dict, required
        """
        self.config = config
        self.logger = logging.getLogger('rlt_logger')
        self.data_exporter = DataExporter(self.config)

    def get_uncertain_samples(self, num_samples, ops=(True, True, True, True)):
        """
        Gets a sample in the alpha uncertainty region of the expected workload

        :param expected_workload:
        :param alpha:
        :param N:
        :return samples:
        """
        np.random.seed(0)

        samples = []

        self.logger.info(
            f'Sampling workloads '
            f'{[op for op, mask in list(zip(["z0", "z1", "q", "w"], ops)) if mask]}')

        # Number of workloads in the sample outside uncertainty region
        for _ in range(num_samples):
            w_hat = np.random.randint(100, size=len(ops))
            w_hat = [num if mask else 0 for num, mask in list(zip(w_hat, ops))]
            w_hat = w_hat / np.sum(w_hat)
            w_hat = np.around(w_hat, PRECISION)
 
            z0_sample, z1_sample, q_sample, w_sample = w_hat
            sample = {}
            sample['z0'] = z0_sample
            sample['z1'] = z1_sample
            sample['q'] = q_sample
            sample['w'] = w_sample
            samples.append(sample)

        return samples

    def run(self, ops=(True, True, True, True)):
        """
        Runs the job
        """
        np.random.seed(0)

        expected_workloads = self.config['expected_workloads']
        N = self.config['uncertain_workload_config']['N']

        all_workloads = []
        config_workload = {}
        config_workload['N'] = N
        config_workload['workloads'] = []

        for w in expected_workloads:
            workload = {}
            workload['expected'] = w
            workload['samples'] = self.get_uncertain_samples(N, ops)
            config_workload['workloads'].append(workload)

        all_workloads.append(config_workload)
        filename = self.config['uncertain_workload_config']['filename']
        self.data_exporter.export_dill_file(all_workloads, filename)

        return all_workloads
