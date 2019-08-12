import os

import cv2
import numpy as np


class CalculationDataset(object):
    """Calculate dataset mean / var """
    def __init__(self, root, array_type, channels, log_interval):
        self.dataset_root = root
        self.array_type = array_type
        self.log_interval = np.array(log_interval, dtype=self.array_type)
        self.img_paths = self.get_paths()
        self.img_paths.sort()
        self.dataset_size = len(self.img_paths)
        self.calculate_mean(channels)
        self.calculate_std(channels)

    def get_paths(self):
        img_paths = []
        for pwd, dirs, files in os.walk(self.dataset_root):
            if len(dirs) == 0:
                for file in files:
                    img_paths.append(os.path.join(pwd, file))
        return img_paths

    def calculate_mean(self, channels):
        if channels == 1:
            self.calculate_mean1()
        elif channels == 3:
            self.calculate_mean3()

    def calculate_std(self, channels):
        if channels == 1:
            self.calculate_std1()
        elif channels == 3:
            self.calculate_std3()

    def calculate_mean1(self):
        precision = self.array_type
        sum_count = 0
        data_mean = 0.0
        dataset_mean = 0.0
        for i, img_path in enumerate(self.img_paths):
            img = np.array(cv2.imread(img_path, 0), dtype=precision)
            img_size = np.array(img.shape[0] * img.shape[1], dtype=precision)
            channel_mean = np.sum(img) / img_size
            data_mean += channel_mean
            if (i+1) % self.log_interval == 0:
                print("{:07d} / {:07d}".format(i+1, self.dataset_size))
                dataset_mean += data_mean / self.log_interval
                sum_count += 1
                data_mean = 0.0
        rest = self.dataset_size % self.log_interval
        if rest != 0:
            dataset_mean += data_mean / rest
            sum_count += 1
        dataset_mean /= sum_count
        self.mean = dataset_mean
        print("mean : {:.4f}".format(dataset_mean / 255.0))

    def calculate_mean3(self):
        precision = self.array_type
        sum_count = 0
        data_mean = np.zeros(3, dtype=precision)
        dataset_mean = np.zeros(3, dtype=precision)
        for i, img_path in enumerate(self.img_paths):
            img = np.array(cv2.imread(img_path), dtype=precision)
            img_size = np.array(img.shape[0] * img.shape[1], dtype=precision)
            channel_mean = img.sum(axis=(0, 1)) / img_size
            data_mean += channel_mean
            if (i+1) % self.log_interval == 0:
                print("{:07d} / {:07d}".format(i+1, self.dataset_size))
                dataset_mean += data_mean / self.log_interval
                sum_count += 1
                data_mean = np.zeros(3, dtype=precision)
        rest = self.dataset_size % self.log_interval
        if rest != 0:
            dataset_mean += data_mean / rest
            sum_count += 1
        dataset_mean /= sum_count
        self.mean = dataset_mean
        print("meanR: {2:.4f}, meanG: {1:.4f}, meanB: {0:.4f}".format(
              *(dataset_mean / 255.0)))

    def calculate_std1(self):
        precision = self.array_type
        sum_count = 0
        data_var = 0.0
        dataset_var = 0.0
        for i, img_path in enumerate(self.img_paths):
            img = np.array(cv2.imread(img_path, 0), dtype=precision)
            img_size = img.shape[0] * img.shape[1]
            channel_var = np.sum((img - self.mean) ** 2) / img_size
            data_var += channel_var
            if (i+1) % self.log_interval == 0:
                print("{:07d} / {:07d}".format(i+1, self.dataset_size))
                dataset_var += data_var / self.log_interval
                sum_count += 1
                data_var = 0.0
        rest = self.dataset_size % self.log_interval
        if rest != 0:
            dataset_var += data_var / rest
            sum_count += 1
        dataset_var /= sum_count
        print("std : {:.4f}".format(dataset_var**0.5 / 255.0))

    def calculate_std3(self):
        precision = self.array_type
        sum_count = 0
        data_var_b, data_var_g, data_var_r = np.zeros(3, dtype=precision)
        dataset_var_b, dataset_var_g, dataset_var_r = np.zeros(3, dtype=precision)
        for i, img_path in enumerate(self.img_paths):
            img = np.array(cv2.imread(img_path), dtype=precision)
            img_size = np.array(img.shape[0] * img.shape[1], dtype=precision)
            var_b = np.sum((img[:, :, 0] - self.mean[0]) ** 2) / img_size
            var_g = np.sum((img[:, :, 1] - self.mean[1]) ** 2) / img_size
            var_r = np.sum((img[:, :, 2] - self.mean[2]) ** 2) / img_size
            data_var_b += var_b
            data_var_g += var_g
            data_var_r += var_r
            if (i+1) % self.log_interval == 0:
                print("{:07d} / {:07d}".format(i+1, self.dataset_size))
                dataset_var_b += data_var_b / self.log_interval
                dataset_var_g += data_var_g / self.log_interval
                dataset_var_r += data_var_r / self.log_interval
                sum_count += 1
                data_var_b, data_var_g, data_var_r = np.zeros(3, dtype=precision)
        rest = self.dataset_size % self.log_interval
        if rest != 0:
            dataset_var_b += data_var_b / rest
            dataset_var_g += data_var_g / rest
            dataset_var_r += data_var_r / rest
            sum_count += 1
        dataset_var_b /= sum_count
        dataset_var_g /= sum_count
        dataset_var_r /= sum_count
        print("stdR: {:.4f}, stdG: {:.4f}, stdB: {:.4f}".format(
              dataset_var_r**0.5 / 255.0,
              dataset_var_g**0.5 / 255.0,
              dataset_var_b**0.5 / 255.0))

class VOC(CalculationDataset):
    def __init__(self, root, array_type, channels, log_interval, year):
        self.year = year
        super().__init__(root, array_type, channels, log_interval)
        

    def get_paths(self):
        imgs = []
        if type(self.year) is list:
            for year in self.year:
                id_txt = os.path.join(self.dataset_root, year,
                                      'ImageSets', 'Main', 'train.txt')
                with open(id_txt, 'r') as f:
                    file_ids = [i.replace('\n', '') for i in f.readlines()]
                    for file_id in file_ids:
                        imgs.append(os.path.join(self.dataset_root, year,
                                    'JPEGImages', file_id+'.jpg'))
        else:
            year = self.year
            id_txt = os.path.join(self.dataset_root, year,
                                  'ImageSets', 'Main', 'train.txt')
            with open(id_txt, 'r') as f:
                file_ids = [i.replace('\n', '') for i in f.readlines()]
                for file_id in file_ids:
                    imgs.append(os.path.join(self.dataset_root, year,
                                'JPEGImages', file_id+'.jpg'))
        return imgs
