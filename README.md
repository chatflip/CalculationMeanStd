# CalculationMeanStd
Calculate datasets of mean and standard deviation

## 1channel dataset
| dataset | mean | std |
|:-------:|:----:|:---:|
| MNIST(train) | 0.1307 | 0.3081 |
| FashionMNIST(train) | 0.2860 | 0.3530 |
| Omniglot(images_background) | 0.9220 | 0.2681 |

## 3channel dataset
| dataset | mean(R, G, B) | std(R, G, B) |
|:-------:|:-------------:|:------------:|
| CIFAR10(train) | (0.4914, 0.4822, 0.4465) | (0.2370, 0.2435, 0.2616) |
| CIFAR100(train) | (0.5071, 0.4865, 0.4409) | (0.673, 0.2564, 0.2762) |
| Caltech101(all images) | (0.5487, 0.5313, 0.5050) | (0.3205, 0.3152, 0.3273) |
| Caltech256(all images) | (, , ) | (, , ) |
| ilsvrc2012(train) | (0.4812, 0.4575, 0.4079) | (0.2832, 0.2761, 0.2895) |
| places365(train 256×256) | (0.4578, 0.4414, 0.4078) | (0.2692, 0.2670, 0.2851) |
| Pascal VOC 2007(train) | (0.4472, 0.4231, 0.3912) | (0.2750, 0.2720, 0.2845) |
| Pascal VOC 2012(train) | (0.4570, 0.4382, 0.4062) | (0.2748, 0.2720, 0.2854) |
| Pascal VOC 0712(train) | (0.4541, 0.4336, 0.4016) | (0.2749, 0.2721, 0.2852) |

# original ilsvrc
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]