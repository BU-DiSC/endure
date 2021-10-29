"""
This class implements data provider for reading various data
"""
import os
import logging
import yaml
import pandas as pd
import dill


class DataProvider(object):
    """
    This class implements the data provider for reading various data.
    """

    def __init__(self, config):
        """Constructor

        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("rlt_logger")

    @classmethod
    def read_config(cls, config_yaml_path):
        """Reads config file

        :param config_yaml_path
        """
        with open(config_yaml_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return config

    def read_csv(self, filename, **kwargs):
        """Reads csv files

        :param filename:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        csv_path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(csv_path,
                         header=0,
                         index_col=False,
                         **kwargs)

        return df

    def read_dill(self, filename):
        """Read dill files

        :param filename:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        dill_path = os.path.join(DATA_DIR, filename)

        with open(dill_path, 'rb') as f:
            data = dill.load(f)

        return data
