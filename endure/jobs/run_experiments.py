"""
This class runs the experiments
"""
import logging
from experiments.experiment_01 import Experiment01
from experiments.experiment_02 import Experiment02
from experiments.experiment_03 import Experiment03
from experiments.experiment_04 import Experiment04
from experiments.experiment_05 import Experiment05


class ExperimentDriver(object):
    """
    Creates experiment driver
    """

    def __init__(self, config):
        """
        Constructor
        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("rlt_logger")

    def run(self):
        self.logger.info("Starting experiments")

        # Get experiment list
        expt_list = self.config['experiments']['expt_list']

        # Run experiments
        for expt_name in expt_list:
            if expt_name == "experiment_01":
                expt = Experiment01(self.config)
                expt.run()
            if expt_name == "experiment_02":
                expt = Experiment02(self.config)
                expt.run()
            if expt_name == "experiment_03":
                expt = Experiment03(self.config)
                expt.run()
            if expt_name == "experiment_04":
                expt = Experiment04(self.config)
                expt.run()
            if expt_name == "experiment_05":
                expt = Experiment05(self.config)
                expt.run()

        self.logger.info("Finished experiments")
