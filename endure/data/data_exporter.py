"""
This class implements data exporter for storing various data
"""
import logging
import dill
import os


class DataExporter(object):
    """
    Data Exporter
    """

    def __init__(self, config):
        """
        Constructor

        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("rlt_logger")

    def export_csv_file(self, df, filename):
        """
        Exports a dataframe in form of a csv file

        :param df:
        :param filename:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, sep=',', header=True, index=False)
        self.logger.info("Exported dataframe to {}".format(filepath))

    def export_dill_file(self, data, filename):
        """
        Exports data in form of a dill file

        :param data:
        :param filename:
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'wb') as f:
            dill.dump(data, f)
        self.logger.info("Exported data to {}".format(filepath))

    def export_figure(self, fig, figname, **kwargs):
        """
        Exports a figure file

        :param fig: Figure handle
        :param figname: Name of the figure with extension
        """
        DATA_DIR = self.config['app']['DATA_DIR']
        filepath = os.path.join(DATA_DIR, figname, **kwargs)
        fig.savefig(filepath)
        self.logger.info("Exported data to {}".format(filepath))
