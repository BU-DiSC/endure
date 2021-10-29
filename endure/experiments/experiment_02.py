"""
Experiment 02
Measures the cumulative I/O performance of a set of workloads for a pair of tunings (robust and nominal)

"""

import logging
import ast

import numpy as np
import pandas as pd
from tqdm import tqdm

from lsm_tree.PyRocksDB import RocksDB
from data.data_provider import DataProvider
from data.data_exporter import DataExporter

class Experiment02(object):

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('rlt_logger')
        self.dp = DataProvider(config)
        self.de = DataExporter(config)

    def run(self):
        num_queries = 100000
        sample_wls = self.dp.read_csv('experiment_02_wls.csv')

        grouped_wls = sample_wls.groupby(['w', 'robust_rho'])
        df = []
        for idx, group in enumerate(grouped_wls):
            _, group = group
            group = group.copy().reset_index(drop=True)
            self.logger.info(f'Workload-Rho Group ({idx + 1} / {len(grouped_wls)})')

            for mode in ['nominal', 'robust']:
                settings = {}
                settings['db_name'] = 'exp02_db'
                settings['path_db'] = self.config['app']['DATABASE_PATH']
                settings['N'], settings['M'] = group[['N', 'M']].iloc[0]
                settings['T'] = group[f'{mode}_T'].iloc[0]
                settings['h'] = group[f'{mode}_m_filt'].iloc[0] / settings['N']
                settings['is_leveling_policy'] = group[f'{mode}_is_leveling_policy'].iloc[0]
                settings['E'] = self.config['lsm_tree_config']['E']

                db = RocksDB(self.config)
                db.init_database(**settings)

                measured_performance = []
                for idx, row in group.iterrows():
                    z0 = int(np.ceil(num_queries * row['z0_s']))
                    z1 = int(np.ceil(num_queries * row['z1_s']))
                    q = int(np.ceil(num_queries * row['q_s']))
                    w = int(np.ceil(num_queries * row['w_s']))

                    self.logger.info(
                        f'Workload ({idx + 1} / {group.shape[0]}) : '
                        f'(z0: {z0}, z1: {z1}, q: {q}, w: {w}) on {mode.upper()}')

                    results = db.run(z0, z1, q, w, prime=10000)
                    named_results = {}
                    for key, val in results.items():
                        self.logger.info(f'{key} : {val}')
                        named_results[f'{mode}_{key}'] = val
                    measured_performance.append(named_results)

                group = pd.concat([group, pd.DataFrame(measured_performance)], axis=1)
                db.delete_database()
                del db

            df.append(group)
            self.de.export_csv_file(pd.concat(df), 'experiment_02_checkpoint.csv')
        df = pd.concat(df)

        self.logger.info('Exporting data from experiment 02')
        self.de.export_csv_file(df, 'experiment_02.csv')
        self.logger.info('Finished experiment 02\n')

        return 0
