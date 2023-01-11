# Data-dependent-stability-of-AT
This is the official repository for "Data-Dependent Stability Analysis of Adversarial Training".  
## Running experiments:  
1. Training the crfat model on CIFAR10(train set and test test together) for generating hypocritical and adversarial poisons.  
  ```python -u craft.py --train_loss AT --epochs 10 --dataset CIFAR10```  
  ```python -u craft.py --train_loss ST --epochs 150 --dataset CIFAR10```
2. Generate EM, REM, HYP, ADV, RANDOM poisoned dataset.  
  ```python -u poison.py  --dataset CIFAR10 --num_classes 10 --poison_type Hyp --poison_aug --craft_model_loss AT  --craft_model_epoch 10```    
  `python -u poison.py  --dataset CIFAR10 --num_classes 10 --poison_type Adv`  
  `python -u poison.py  --dataset CIFAR10 --num_classes 10 --poison_type Random`     
3. Adversarial training on poisoned dataset.
  `python -u at_poison.py --dataset CIFAR100 --eps=4 --poison_type Hyp --poison_aug --craft_model_loss AT --craft_model_epoch 10`
  `python -u at_poison.py --dataset CIFAR100 --eps=4 --poison_type Adv`
  `python -u at_poison.py --dataset CIFAR100 --eps=4 --poison_type Random`


