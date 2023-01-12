# Data-dependent-stability-of-AT
This is the official repository for "Data-Dependent Stability Analysis of Adversarial Training".  
## Running experiments:  
1. Training the crfat model on CIFAR10(train set and test test together) for generating hypocritical and adversarial poisons.  
  ```
  python craft.py --train_loss AT --epochs 10 --dataset CIFAR10
  ```     
  ```
  python craft.py --train_loss ST --epochs 150 --dataset CIFAR10
  ```
2. Generate EM, REM, HYP, ADV, RANDOM poisoned dataset.  
  ```
  python poison.py  --dataset CIFAR10 --num_classes 10 --poison_type Hyp  
  --poison_eps 8 --poison_aug --craft_model_loss AT  --craft_model_epoch 10
  ```    
  ```
  python poison.py  --dataset CIFAR10 --num_classes 10 --poison_type Adv  --poison_eps   
  ```      
  ```
  python poison.py  --dataset CIFAR10 --num_classes 10 --poison_type Random  --poison_eps   
  ```         
  ```
  python unlearnable/generate_em.py  --dataset CIFAR10 --pgd-random-start  
  --pgd-radius 8 --pgd-step-size 1.6   
  --save-dir ../exp_data/cifar10/em8 --save-name em
  ```
  ```
  python unlearnable/generate_robust_em.py  --dataset CIFAR10 --pgd-random-start 
  --atk-pgd-random-start --atk-pgd-radius 4 --atk-pgd-step-size 0.8    
  --save-dir ../exp_data/cifar100/rem8-4 --save-name=rem
  ```
3. Adversarial training on poisoned dataset.    
  ```
  python at_poison.py --dataset CIFAR10 --eps=4 --poison_type Hyp   
  --poison_eps 8 --poison_aug --craft_model_loss AT --craft_model_epoch 10
  ```    
  ```
  python at_poison.py --dataset CIFAR10 --eps=4 --poison_type Adv --poison_eps  
  ```   
  ```
  python at_poison.py --dataset CIFAR10 --eps=4 --poison_type Random --poison_eps  
  ```      
  ```
  python at_unlearnable.py --dataset CIFAR10 --eps=4 --poison_type em
  ```
   ```
  python at_unlearnable.py --dataset CIFAR10 --eps=4 --poison_type rem
  ```
  

## Results
## Acknowledgment
