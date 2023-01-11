# Data-dependent-stability-of-AT
This is the official repository for "Data-Dependent Stability Analysis of Adversarial Training".
Running experiments:
1. training the crfat model on CIFAR10(train set and test test together) for generating hypocritical and adversarial poisons.
```python -u craft.py --train_loss AT --epochs 10 --dataset CIFAR10```
