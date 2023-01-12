# Data-dependent-stability-of-AT
This is the official repository for "Data-Dependent Stability Analysis of Adversarial Training". 

efwegerwg
## Requirements:  
* Python 3.7.10 
* PyTorch 1.13.1
* Torchvision 0.14.1

## Running experiments:  
We give an example of creating different poisons from CIFAR-10 dataset, CIFAR-100 is similar.  
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
  python poison.py  --dataset CIFAR10 --num_classes 10 --poison_type Adv  --poison_eps 8   
  ```      
  ```
  python poison.py  --dataset CIFAR10 --num_classes 10 --poison_type Random  --poison_eps 8  
  ```         
  ```
  python unlearnable/generate_em.py  --dataset CIFAR10 --pgd-random-start  
  --pgd-radius 8 --pgd-step-size 1.6   
  --save-dir ./exp_data/cifar10/em8 --save-name em
  ```
  ```
  python unlearnable/generate_robust_em.py  --dataset CIFAR10    
  --pgd-random-start --pgd-radius 8 --pgd-step-size 1.6   
  --atk-pgd-random-start --atk-pgd-radius 4 --atk-pgd-step-size 0.8    
  --save-dir ./exp_data/cifar10/rem8-4 --save-name=rem
  ```
3. Adversarial training on poisoned dataset.    
  ```
  python at_poison.py --dataset CIFAR10 --eps=4 --poison_type Hyp   
  --poison_eps 8 --poison_aug --craft_model_loss AT --craft_model_epoch 10
  ```    
  ```
  python at_poison.py --dataset CIFAR10 --eps=4 --poison_type Adv --poison_eps 8     
  ```   
  ```
  python at_poison.py --dataset CIFAR10 --eps=4 --poison_type Random --poison_eps 8      
  ```      
  ```
  python at_unlearnable.py --dataset CIFAR10 --eps=4 --poison_type em
  ```
   ```
  python at_unlearnable.py --dataset CIFAR10 --eps=4 --poison_type rem
  ```
  

## Results
## Acknowledgment
Robust Unlearnable Example and Unlearnable Example:  
https://github.com/fshp971/robust-unlearnable-examples   
Hypocritical Perturbation:  
https://github.com/TLMichael/Hypocritical-Perturbation   
