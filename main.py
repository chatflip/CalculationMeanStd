import time

import numpy as np

from CalculationMeanStd import CalculationDataset, VOC

if __name__ == "__main__":
    starttime = time.time()
    """
    dataset_root = "path/to/dataset"
    channels = 3
    log_interval = 10000
    cal = CalculationDataset(dataset_root, np.float64, channels, log_interval)
    """
    dataset_root = "data/VOCdevkit"
    channels = 3
    log_interval = 10000
    year = ['VOC2007', 'VOC2012']
    cal = VOC(dataset_root, np.float64, channels, log_interval, year)
    endtime = time.time()
    interval = endtime - starttime
    print("elapsed time = %dh %dm %ds" % (int(interval / 3600),
                                          int(interval % 3600 / 60),
                                          int(interval % 3600 % 60)))
