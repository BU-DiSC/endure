"""
Python API for RocksDB
"""
import logging
import os
import re
from pprint import pprint
import shutil
import subprocess
import numpy as np
from pkg_resources import resource_listdir

THREADS = 4

class RocksDB(object):
    """
    Python API for RocksDB
    """
    def __init__(self, config):
        """
        Constructor

        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
        self.level_hit_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] \(l0, l1, l2plus\) : '
            r'\((-?\d+), (-?\d+), (-?\d+)\)'
        )
        self.bf_count_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] \(bf_true_neg, bf_pos, bf_true_pos\) : '
            r'\((-?\d+), (-?\d+), (-?\d+)\)'
        )
        self.compaction_bytes_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] \(bytes_written, compact_read, compact_write, flush_write\) : '
            r'\((-?\d+), (-?\d+), (-?\d+), (-?\d+)\)'
        )
        self.block_read_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] \(block_read_count\) : '
            r'\((-?\d+)\)'
        )
        self.time_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] \(z0, z1, q, w\) : '
            r'\((-?\d+), (-?\d+), (-?\d+), (-?\d+)\)'
        )
        self.compact_time_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] \(remaining_compactions_duration\) : '
            r'\((-?\d+)\)'
        )
        self.runs_per_level_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] runs_per_level : '
            r'(\[[0-9,\s]+\])'
        )
        self.existing_keys_prog = re.compile(r'\[[0-9:.]+\]\[info\] Writing out ([0-9]+) existing keys')

    def options_from_config(self):
        db_settings = {}
        db_settings['path_db'] = self.config['app']['DATABASE_PATH']
        db_settings['N'] = self.config['lsm_tree_config']['N']
        db_settings['B'] = self.config['lsm_tree_config']['B']
        db_settings['E'] = self.config['lsm_tree_config']['E']
        db_settings['M'] = self.config['lsm_tree_config']['M']
        db_settings['P'] = self.config['lsm_tree_config']['P']
        db_settings['is_leveling_policy'] = self.config['lsm_tree_config']['is_leveling_policy']

        # Defaults
        db_settings['db_name'] = 'default'
        db_settings['h'] = 5
        db_settings['T'] = 10

        return db_settings

    def estimate_levels(self):
        mbuff = self.M - (self.h * self.N)
        l = np.ceil((np.log((self.N * self.E) / mbuff) + 1) / np.log(self.T))

        return l

    def init_database(self, db_name, path_db, h, T, N, E, M, is_leveling_policy=True, destroy=True, bulk_stop_early=False, **kwargs):
        """[summary]

        :param db_name: database name
        :param path_db: path to the database
        :param h: bits per element for bloom filters
        :param T: Size ration
        :param N: Total elements
        :param B: Number entries that fit in a disk page
        :param E: Size of the entry in bits
        :param M: Total memory
        :param is_leveling_policy: Tiering vs Leveling, defaults to True
        :param destroy: Destroy DB in fodler if already exists, defaults to True

        :return existing_keys: Total number of keys in the DB
        """
        self.path_db = path_db
        self.db_name = db_name
        self.h, self.T = h, int(np.ceil(T))
        self.N, self.M = int(N), int(M)
        self.E = (E >> 3) # Converts bits -> bytes
        self.is_leveling_policy = is_leveling_policy
        self.destroy = destroy

        os.makedirs(os.path.join(self.path_db, self.db_name), exist_ok=True)

        mbuff = int(self.M - (self.h * self.N)) >> 3

        self.K = 1 if self.is_leveling_policy else self.T - 1
        self.Z = self.K

        policy = 'Level' if is_leveling_policy else 'Tier'
        db_dir = os.path.join(self.path_db, self.db_name)
        self.logger.info('Creating DB {}'.format(self.db_name))
        self.logger.info(f'T: {self.T}, bpe: {self.h:.2f}, policy: {policy}, mbuff: {mbuff}, E: {self.E}, N: {self.N}, M: {self.M}')
        cmd = [
            self.config['app']['BUILDER_PATH'],
            db_dir,
            '-d',
            '-T {}'.format(self.T),
            '-K {}'.format(self.K),
            '-Z {}'.format(self.Z),
            '-B {}'.format(mbuff),
            '-E {}'.format(self.E),
            '-b {:.2f}'.format(self.h),
            '-N {}'.format(self.N),
            '--parallelism {}'.format(8),
            '--key-file {}'.format(self.config['app']['KEY_FILE_PATH'])
        ]
        if bulk_stop_early:
            cmd += ['--early_fill_stop']
        cmd = ' '.join(cmd)
        self.logger.debug(f'{cmd}')

        completed_process, _ = subprocess.Popen(
            cmd,
            # stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
        ).communicate()

        existing_keys = int(self.existing_keys_prog.search(completed_process).groups()[0])

        return existing_keys

    def create_temp_copy(self, tmp_folder):
        """
        Creates a copy of the DB in the /tmp folder of a linux system

        :param tmp_folder: name of the temporarly folder
        :type tmp_folder: str. required
        """
        os.makedirs(tmp_folder, exist_ok=True)
        shutil.copytree(os.path.join(self.path_db, self.db_name), tmp_folder, dirs_exist_ok=True)

    def delete_temp_copy(self, tmp_folder):
        """
        Deletes the temporary DB copy

        :param tmp_folder: path to the temporary DB
        :type tmp_folder: str, required
        """
        shutil.rmtree(tmp_folder)

    def delete_database(self):
        """
        Deletes the database
        """
        db_dir = os.path.join(self.path_db, self.db_name)
        shutil.rmtree(db_dir)

    def run(self, num_z0, num_z1, num_q, num_w, prime=10000, copy=False):
        """
        Runs a set of queries on the database

        :param num_z0: empty reads
        :param num_z1: non-empty reads
        :param num_w: writes
        """
        if copy:
            db_dir = os.path.join(self.path_db, self.db_name + '_tmp')
            self.create_temp_copy(db_dir)
        else:
            db_dir = os.path.join(self.path_db, self.db_name)

        cmd = [
            self.config['app']['EXECUTION_PATH'],
            db_dir,
            f'-e {num_z0}',
            f'-r {num_z1}',
            f'-q {num_q}',
            f'-w {num_w}',
            f'-p {prime}',
            '--parallelism {}'.format(THREADS),
            '--key-file {}'.format(self.config['app']['KEY_FILE_PATH'])
        ]
        cmd = ' '.join(cmd)
        self.logger.debug(f'{cmd}')

        proc = subprocess.Popen(
            cmd,
            # stdin=None,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
        )

        results = {}

        try:
            timeout = 10 * 60 * 60
            proc_results, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.logger.warn('Timeout limit reached. Aborting')
            proc.kill()
            results['l0_hit'] = 0 
            results['l1_hit'] = 0 
            results['l2_plus_hit'] = 0 
            results['z0_ms'] = 0 
            results['z1_ms'] = 0 
            results['q_ms'] = 0 
            results['w_ms'] = 0 
            results['filter_neg'] = 0 
            results['filter_pos'] = 0 
            results['filter_pos_true'] = 0 
            results['bytes_written'] = 0 
            results['compact_read'] = 0 
            results['compact_write'] = 0 
            results['flush_written'] = 0 
            results['blocks_read'] = 0 
            results['runs_per_level'] = 0 
            if copy:
                self.delete_temp_copy(db_dir)
            return results 

        level_hit_results = [int(result) for result in self.level_hit_prog.search(proc_results).groups()]
        bf_count_results = [int(result) for result in self.bf_count_prog.search(proc_results).groups()]
        compaction_results = [int(result) for result in self.compaction_bytes_prog.search(proc_results).groups()]
        block_read_result = [int(result) for result in self.block_read_prog.search(proc_results).groups()]
        compact_time_result = [int(result) for result in self.compact_time_prog.search(proc_results).groups()]
        time_results = [int(result) for result in self.time_prog.search(proc_results).groups()]
        runs_per_level = self.runs_per_level_prog.findall(proc_results)[0]

        if copy:
            self.delete_temp_copy(db_dir)

        results['l0_hit'] = level_hit_results[0]
        results['l1_hit'] = level_hit_results[1]
        results['l2_plus_hit'] = level_hit_results[2]

        results['z0_ms'] = time_results[0]
        results['z1_ms'] = time_results[1]
        results['q_ms'] = time_results[2]
        results['w_ms'] = time_results[3]
        results['compact_ms'] = compact_time_result[0]

        results['filter_neg'] = bf_count_results[0]
        results['filter_pos'] = bf_count_results[1]
        results['filter_pos_true'] = bf_count_results[2]

        results['bytes_written'] = compaction_results[0]
        results['compact_read'] = compaction_results[1]
        results['compact_write'] = compaction_results[2]
        results['flush_written'] = compaction_results[3]

        results['blocks_read'] = block_read_result[0]

        results['runs_per_level'] = runs_per_level.strip()

        return results 
