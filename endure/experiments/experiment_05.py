"""
Experiment 03
Writes and Reads hybrid experiment on PyRocksDB

"""

import logging
import ast
from copy import deepcopy
from pprint import pprint

import numpy as np
import pandas as pd
from scipy.special import rel_entr
from tqdm import tqdm

from lsm_tree.PyRocksDB import RocksDB
from data.data_provider import DataProvider
from data.data_exporter import DataExporter
from jobs.sample_uncertain_workloads import SampleUncertainWorkloads
from robust.workload_uncertainty import WorkloadUncertainty
from lsm_tree.nominal import NominalWorkloadTuning
from lsm_tree.cost_function import CostFunction

class Experiment05(object):

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('rlt_logger')
        self.dp = DataProvider(config)
        self.de = DataExporter(config)

    def create_tunings(self, wl_idxs, wl_rhos, expected_wls,
                       db_sizes, bpe, buffer_min):
        lsm_config = deepcopy(self.config['lsm_tree_config'])

        # Create tunings for respective expected workloads
        tunings = []
        for wl_idx, rho in zip(wl_idxs, wl_rhos):
            for N in db_sizes:
                design = {}
                design['workload_idx'] = wl_idx
                design['z0'] = expected_wls[wl_idx]['z0']
                design['z1'] = expected_wls[wl_idx]['z1']
                design['q'] = expected_wls[wl_idx]['q']
                design['w'] = expected_wls[wl_idx]['w']
                design['N'] = lsm_config['N'] = N
                design['phi'] = lsm_config['phi']
                design['B'] = lsm_config['B']
                design['s'] = lsm_config['s']
                design['E'] = lsm_config['E']
                design['M'] = lsm_config['M'] = ((bpe * lsm_config['N'])
                                                 + buffer_min)

                cf = CostFunction(**lsm_config, **expected_wls[wl_idx])
                nominal = NominalWorkloadTuning(cf)
                nominal_design = nominal.get_nominal_design()
                design['nominal_m_filt'] = nominal_design['M_filt']
                design['nominal_m_buff'] = nominal_design['M_buff']
                design['nominal_T'] = nominal_design['T']
                design['nominal_cost'] = nominal_design['cost']
                design['nominal_bpe'] = nominal_design['M_filt'] / N
                design['nominal_is_leveling_policy'] = nominal_design['is_leveling_policy']

                robust = WorkloadUncertainty(cf)
                design['rho'] = rho
                tiering_design = robust.get_robust_tiering_design(
                    rho, nominal_design=None)
                leveling_design = robust.get_robust_leveling_design(
                    rho, nominal_design=None)
                if tiering_design['obj'] < leveling_design['obj']:
                    robust_design = tiering_design
                else:
                    robust_design = leveling_design
                design['robust_exit_mode'] = robust_design['exit_mode']
                design['robust_m_filt'] = robust_design['M_filt']
                design['robust_m_buff'] = robust_design['M_buff']
                design['robust_T'] = robust_design['T']
                design['robust_cost'] = robust_design['cost']
                design['robust_is_leveling_policy'] = robust_design['is_leveling_policy']
                design['robust_bpe'] = robust_design['M_filt'] / N

                # Append the design to the dataframe
                tunings.append(deepcopy(design))

        return tunings

    def create_sessions(self, df, samples):
        sessions = []
        session = df[df.z0_s + df.z1_s > 0.8].sample(samples, replace=False, random_state=0)
        session['session_id'] = 0
        sessions.append(session)

        session = df[df.q_s > 0.8].sample(samples, replace=False, random_state=0)
        session['session_id'] = 1
        sessions.append(session)

        session = df[df.z0_s > 0.8].sample(samples, replace=False, random_state=0)
        session['session_id'] = 2
        sessions.append(session)

        session = df[df.z1_s > 0.8].sample(samples, replace=False, random_state=0)
        session['session_id'] = 3
        sessions.append(session)

        session = df[df.w_s > 0.8].sample(samples, replace=False, random_state=0)
        # session = df[df.z0_s + df.z1_s > 0.8].sample(samples, replace=False, random_state=0)
        session['session_id'] = 4
        sessions.append(session)

        # session = df[df.z0_s + df.z1_s > 0.8].sample(samples, replace=False, random_state=0)
        replace = len(df[df.dist < 0.2]) < samples
        session = df[df.dist < 0.2].sample(samples, replace=replace, random_state=0)
        session['session_id'] = 5
        sessions.append(session)

        return pd.concat(sessions, ignore_index=True)

    def run(self):
        db_sizes = [1e6, 3e6, 1e7, 3e7, 1e8, 2e8]
        # db_sizes = [1e6, 2e6, 4e6, 1e7, 3e7]
        expected_wls = [
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
        wl_idxs = [7, 11, 16]
        wl_rhos = [0.25, 0.25, 0.5]
        op_mask = (True, True, True, True)
        bpe = 10
        buffer_min = 1 * 1024 * 1024 * 8 # 1 MiB in bits

        suw = SampleUncertainWorkloads(self.config)
        sample_wls = suw.get_uncertain_samples(10000, op_mask)

        self.logger.info('Creating tunings')
        tunings = self.create_tunings(
            wl_idxs, wl_rhos, expected_wls,
            db_sizes, bpe, buffer_min)

        tables = []
        for design in tunings:
            sample_wls_dist = []
            row = []
            w0 = [design['z0'], design['z1'], design['q'], design['w']]
            self.logger.info(f'RUNNING DESIGN FOR WORKLOAD {w0}')
            for idx, wl in enumerate(sample_wls):
                w_hat = [wl['z0'], wl['z1'], wl['q'], wl['w']]
                sample_wls_dist.append(
                    {'sample_idx' : idx,
                     'z0_s' : wl['z0'],
                     'z1_s' : wl['z1'],
                     'q_s' : wl['q'],
                     'w_s' : wl['w'],
                     'dist' : np.sum(rel_entr(w_hat, w0))})

            sample_wls_dist = pd.DataFrame(sample_wls_dist)
            sessions = self.create_sessions(sample_wls_dist, 5)

            table = []
            for idx, wl in sessions.iterrows():
                row = deepcopy(design)
                row['sample_idx'] = wl['sample_idx']
                row['num_queries'] = (design['N'] * 0.001)
                # row['num_queries'] = 100000 
                row['z0_s'] = wl['z0_s']
                row['z1_s'] = wl['z1_s']
                row['q_s'] = wl['q_s']
                row['w_s'] = wl['w_s']
                row['kl_div'] = wl['dist']
                row['session_id'] = wl['session_id']
                table.append(row)
            table = pd.DataFrame(table)

            for mode in ['nominal', 'robust']:
                settings = {}
                settings['db_name'] = 'exp05_db'
                settings['path_db'] = self.config['app']['DATABASE_PATH']
                settings['N'] = design['N']
                settings['M'] = design['M']
                settings['E'] = design['E']
                settings['T'] = design[f'{mode}_T']
                settings['h'] = design[f'{mode}_bpe']
                settings['is_leveling_policy'] = design[f'{mode}_is_leveling_policy']

                db = RocksDB(self.config)
                existing_keys = db.init_database(**settings)
                self.logger.info(f'Loaded DB with {existing_keys} keys')

                measured_performance = []
                for idx, wl in sessions.iterrows():
                    z0 = int(np.ceil(row['num_queries'] * wl['z0_s']))
                    z1 = int(np.ceil(row['num_queries'] * wl['z1_s']))
                    q = int(np.ceil(row['num_queries'] * wl['q_s']))
                    w = int(np.ceil(row['num_queries'] * wl['w_s']))
                    self.logger.info(
                        f'Running {design["N"]:.0e}'
                        f' : {mode}'
                        f' : session {(idx + 1):02d} / {sessions.shape[0]}'
                        f' : ({z0}, {z1}, {q}, {w})'
                    )

                    results = db.run(z0, z1, q, w, prime=10000)
                    named_results = {}
                    for key, val in results.items():
                    #     self.logger.info(f'{key} : {val}')
                        named_results[f'{mode}_{key}'] = val
                    measured_performance.append(named_results)

                table = pd.concat([table, pd.DataFrame(measured_performance)], axis=1)
                db.delete_database()
                del db

            tables.append(table)
            self.de.export_csv_file(pd.concat(tables), 'experiment_05_checkpoint.csv')
        df = pd.concat(tables)

        self.logger.info('Exporting data from experiment 05')
        self.de.export_csv_file(df, 'experiment_05.csv')
        self.logger.info('Finished experiment 05')

        return 0

