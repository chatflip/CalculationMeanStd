import time

from CalculationMeanStd import CalculationDataset

if __name__ == "__main__":
    starttime = time.time()
    dataset_root = "path/to/dataset"
    channels = 3
    log_interval = 10000
    cal = CalculationDataset(dataset_root, channels, log_interval)
    cal.calculate_mean()
    cal.calculate_std()
    endtime = time.time()
    interval = endtime - starttime
    print("elapsed time = %dh %dm %ds" % (int(interval / 3600),
                                          int(interval % 3600 / 60),
                                          int(interval % 3600 % 60)))
