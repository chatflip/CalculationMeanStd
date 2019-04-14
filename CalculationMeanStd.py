import os

import cv2
import numpy as np


class CalculationDataset(object):
    """Calculate dataset mean / var """
    def __init__(self, root, channels, log_interval):
        self.dataset_root = root
        self.channels = channels
        self.log_interval = log_interval
        self.img_paths = self.get_paths()
        self.img_paths.sort()
        self.dataset_size = len(self.img_paths)

    def get_paths(self):
        img_paths = []
        for pwd, dirs, files in os.walk(self.dataset_root):
            if len(dirs) == 0:
                for file in files:
                    img_paths.append(os.path.join(pwd, file))
        return img_paths

    def calculate_mean(self):
        if self.channels == 1:
            dataset_mean = 0.0
            for i, img_path in enumerate(self.img_paths):
                img = np.array(cv2.imread(img_path, 0), dtype=np.float64)
                img_size = (img.shape[0] * img.shape[1])
                channel_mean = np.sum(img) / img_size
                dataset_mean += channel_mean / self.dataset_size
                if (i+1) % self.log_interval == 0:
                    print("{:07d} / {:07d}".format(i+1, len(self.img_paths)))
            print("mean : {:.4f}"
                  .format(dataset_mean / 255.0, dataset_mean))
            self.mean = dataset_mean
        elif self.channels == 3:
            dataset_mean_b, dataset_mean_g, dataset_mean_r = 0.0, 0.0, 0.0
            for i, img_path in enumerate(self.img_paths):
                img = np.array(cv2.imread(img_path), dtype=np.float64)
                img_size = img.shape[0] * img.shape[1]
                mean_b = np.sum(img[:, :, 0]) / img_size
                mean_g = np.sum(img[:, :, 1]) / img_size
                mean_r = np.sum(img[:, :, 2]) / img_size
                dataset_mean_b += mean_b / self.dataset_size
                dataset_mean_g += mean_g / self.dataset_size
                dataset_mean_r += mean_r / self.dataset_size
                if (i+1) % self.log_interval == 0:
                    print("{:07d} / {:07d}".format(i+1, self.dataset_size))
            print("meanB: {:.4f}, meanG: {:.4f}, meanR: {:.4f}"
                  .format(dataset_mean_b / 255.0,
                          dataset_mean_g / 255.0,
                          dataset_mean_r / 255.0,))
            self.mean = [dataset_mean_b, dataset_mean_g, dataset_mean_r]

    def calculate_std(self):
        if self.channels == 1:
            dataset_var = 0.0
            for i, img_path in enumerate(self.img_paths):
                img = np.array(cv2.imread(img_path, 0), dtype=np.float64)
                img_size = img.shape[0] * img.shape[1]
                channel_var = np.sum((img - self.mean) ** 2) / img_size
                dataset_var += channel_var / self.dataset_size
                if (i+1) % self.log_interval == 0:
                    print("{:07d} / {:07d}".format(i+1, len(self.img_paths)))
            print("std : {:.4f}, {:.2f}/255"
                  .format(dataset_var**0.5 / 255.0, dataset_var**0.5))
        elif self.channels == 3:
            dataset_var_b, dataset_var_g, dataset_var_r = 0.0, 0.0, 0.0
            for i, img_path in enumerate(self.img_paths):
                img = np.array(cv2.imread(img_path, 0), dtype=np.float64)
                img_size = img.shape[0] * img.shape[1]
                var_b = np.sum((img - self.mean[0]) ** 2) / img_size
                var_g = np.sum((img - self.mean[1]) ** 2) / img_size
                var_r = np.sum((img - self.mean[2]) ** 2) / img_size
                dataset_var_b += var_b / self.dataset_size
                dataset_var_g += var_g / self.dataset_size
                dataset_var_r += var_r / self.dataset_size
                if (i+1) % self.log_interval == 0:
                    print("{:07d} / {:07d}".format(i+1, self.dataset_size))
            print("stdB: {:.4f}, stdG: {:.4f}, stdR: {:.4f}"
                  .format(dataset_var_b**0.5 / 255.0,
                          dataset_var_g**0.5 / 255.0,
                          dataset_var_r**0.5 / 255.0))
