"""
Experiment 04
Scaling database experiment

OUTLINE:
    1. Fix database tuning
    2. Fix workload distribution
    3. Run workload on database tuning over different values of N (# of elment in DB)

"""
import logging

import numpy as np
import pandas as pd

from lsm_tree.PyRocksDB import RocksDB
from lsm_tree.cost_function import CostFunction
from data.data_provider import DataProvider
from data.data_exporter import DataExporter

class Experiment04(object):

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('rlt_logger')
        self.dp = DataProvider(config)
        self.de = DataExporter(config)

    def run(self):
        num_queries = 100000
        workload = (0.25, 0.25, 0.25, 0.25)

        z0 = int(np.ceil(num_queries * workload[0]))
        z1 = int(np.ceil(num_queries * workload[1]))
        q = int(np.ceil(num_queries  * workload[2]))
        w = int(np.ceil(num_queries  * workload[3]))

        num_entries = (1e5, 3e5, 6e5, 1e6, 3e6, 6e6, 1e7, 3e7, 6e7, 1e8, 2e8)

        bpe_budget = 10
        buffer = 8 * 1024 * 1024 * 8 # MiB in bits
        size_ratio = 8

        df = []
        for n in num_entries:
            self.logger.info(f'Building DB at size : {n}')
            row, settings = self.config['lsm_tree_config'].copy(), self.config['lsm_tree_config'].copy()
            settings['db_name'] = 'exp04_db'
            settings['path_db'] = self.config['app']['DATABASE_PATH']
            settings['T'] = row['T'] = size_ratio
            settings['N'] = row['N'] = n
            settings['M'] = row['M'] = buffer + (bpe_budget * n) 
            settings['h'] = row['h'] = bpe_budget
            settings['is_leveling_policy'] = True

            cf = CostFunction(
                settings['N'],
                settings['phi'],
                settings['s'],
                settings['B'],
                settings['E'],
                settings['M'],
                settings['is_leveling_policy'],
                z0, z1, q, w)

            db = RocksDB(self.config)
            _ = db.init_database(**settings, bulk_stop_early=False)

            self.logger.info('Running workload')
            results = db.run(z0, z1, q, w, prime=10000)
            for key, val in results.items():
                self.logger.info(f'{key} : {val}')
                row[f'{key}'] = val

            row['model_io'] = cf.calculate_cost(settings['h'], settings['T'])
            df.append(row)
            self.de.export_csv_file(pd.DataFrame(df), 'experiment_04_checkpoint.csv')

        self.logger.info('Exporting data from experiment 04')
        df = pd.DataFrame(df)
        self.de.export_csv_file(df, 'experiment_04.csv')
        self.logger.info('Finished experiment 04\n')

        return 0
