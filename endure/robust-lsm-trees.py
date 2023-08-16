"""
This class implements the driver program for the robust-lsm-trees
project
"""

import os
import logging
import sys
import yaml
from jobs.create_workload_uncertainty_tunings import CreateWorkloadUncertaintyTunings
from jobs.create_workload_nominal_tunings import CreateNominalWorkloadTunings
from jobs.sample_uncertain_workloads import SampleUncertainWorkloads
from jobs.run_experiments import ExperimentDriver


class RobustLSMTreesDriver(object):
    """
    This class implements the driver program for the robust-lsm-trees
    project.
    """

    def __init__(self, config_):
        """
        Constructor
        :param config_:
        """
        self.config = config_

        # Initialize the logging
        if self.config['app']['app_logging_level'] == 'DEBUG':
            logging_level = logging.DEBUG
        elif self.config['app']['app_logging_level'] == 'INFO':
            logging_level = logging.INFO
        else:
            logging_level = logging.INFO

        logging.basicConfig(
            format="LOG: %(asctime)-15s:[%(filename)s]: %(message)s",
            datefmt='%m/%d/%Y %I:%M:%S %p')

        self.logger = logging.getLogger("rlt_logger")
        self.logger.setLevel(logging_level)

    def run(self):
        """Execute jobs"""
        self.logger.info("Starting app: {}".format(self.config['app']['app_name']))

        # Get job list
        job_list = self.config['jobs']['job_list']

        # Execute jobs
        for job_name in job_list:
            if job_name == "create_workload_uncertainty_tunings":
                job = CreateWorkloadUncertaintyTunings(self.config)
                job.run()
            if job_name == "run_experiments":
                job = ExperimentDriver(self.config)
                job.run()
            if job_name == "sample_uncertain_workloads":
                job = SampleUncertainWorkloads(self.config)
                job.run()
            if job_name == 'create_workload_nominal_tunings':
                job = CreateNominalWorkloadTunings(self.config)
                job.run()

        self.logger.info("Finished")


if __name__ == "__main__":

    # Load configuration
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        dirname = os.path.dirname(__file__)
        config_yaml_path = os.path.join(dirname, 'config/robust-lsm-trees.yaml')

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Start driver
    driver = RobustLSMTreesDriver(config)
    driver.run()
