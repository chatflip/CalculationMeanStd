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
                                          int((interval % 3600) / 60),
                                          int((interval % 3600) % 60)))
# MNIST(train) mean: 0.1307, std: 0.3081
# fashion MNIST(train) mean: 0.2860, std: 0.3530
# omniglot(image_evaluation) mean: 0.9221. std: 0.2681
# CIFAR10(train) mean B: 0.4465, G: 0.4822, R: 0.4914
#                std  B: 0.2414 , G:0.2393, R: 0.2396
# CIFAR100(train) mean B: 0.4409, G: 0.4865, R: 0.5071
#                 std  B : 0.2547, G: 0.2507, R: 0.2516
# caltech101(train) mean B: 0.5050, G: 0.5313, R: 0.5487
#                   std  B: 0.3117, G: 0.3104, R: 0.3107
# caltech256(train) 