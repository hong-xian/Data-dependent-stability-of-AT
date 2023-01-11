# Data-dependent-stability-of-AT
This is the official repository for "Data-Dependent Stability Analysis of Adversarial Training".  
## Running experiments:  
1. Training the crfat model on CIFAR10(train set and test test together) for generating hypocritical and adversarial poisons.  
  ```
  python -u craft.py --train_loss AT --epochs 10 --dataset CIFAR10
  ```     
  ```
  python -u craft.py --train_loss ST --epochs 150 --dataset CIFAR10
  ```
2. Generate EM, REM, HYP, ADV, RANDOM poisoned dataset.  
  ```
  python -u poison.py  --dataset CIFAR10 --num_classes 10 --poison_type Hyp   
  --poison_aug --craft_model_loss AT  --craft_model_epoch 10
  ```    
  ```
  python -u poison.py  --dataset CIFAR10 --num_classes 10 --poison_type Adv
  ```      
  ```
  python -u poison.py  --dataset CIFAR10 --num_classes 10 --poison_type Random
  ```         
  ```
  python -u generate_em.py --pgd-random-start --pgd-radius 8 --pgd-step-size 1.6   
  --save-dir ./exp_data/cifar10/em8 --save-name em
  ```
  ```
  python -u generate_robust_em.py  --dataset CIFAR100 --pgd-random-start --atk-pgd-random-start  
  --save-dir=./revised_exp_data/cifar100/rem8-4 --save-name=rem
  ```
3. Adversarial training on poisoned dataset.    
  ```
  python -u at_poison.py --dataset CIFAR10 --eps=4 --poison_type Hyp   
  --poison_aug --craft_model_loss AT --craft_model_epoch 10
  ```    
  ```
  python -u at_poison.py --dataset CIFAR10 --eps=4 --poison_type Adv
  ```   
  ```
  python -u at_poison.py --dataset CIFAR10 --eps=4 --poison_type Random
  ```      

## Results
## Acknowledgment
