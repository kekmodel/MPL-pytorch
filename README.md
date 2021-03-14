# Meta Pseudo Labels
This is an unofficial PyTorch implementation of [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580).
The official Tensorflow implementation is [here](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels).


## Results

|  | CIFAR-10-4K | SVHN-1K | ImageNet-10% |
|:---:|:---:|:---:|:---:|
| Paper (w/ finetune) | 96.11 ± 0.07 | 98.01 ± 0.07 | 73.89 |
| This code (w/o finetune) | 94.46 | - | - |
| This code (w/ finetune) | WIP | - | - |
| Acc. curve | [link](https://tensorboard.dev/experiment/sRh7ke1jRRWrOFBpC4rhWQ/) | - | - |

* I have experienced some difficulties while reproducing paper's result. 
* Please let me know where to modify my code! ([issue](https://github.com/kekmodel/MPL-pytorch/issues/2))

## Usage

Train the model by 4000 labeled data of CIFAR-10 dataset:

```
python main.py --seed 5 --name cifar10-4K.5 --expand-labels --dataset cifar10 --num-classes 10 --num-labeled 4000 --total-steps 300000 --eval-step 1000 --randaug 2 16 --batch-size 128 --teacher_lr 0.05 --student_lr 0.05 --weight-decay 5e-4 --ema 0.995 --nesterov --mu 7 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 8 --warmup-steps 5000 --uda-steps 5000 --student-wait-steps 3000 --teacher-dropout 0.2 --student-dropout 0.2 --amp
```

Train the model by 10000 labeled data of CIFAR-100 dataset by using DistributedDataParallel:
```
python -m torch.distributed.launch --nproc_per_node 4 main.py --seed 5 --name cifar100-10K.5 --dataset cifar100 --num-classes 100 --num-labeled 10000 --expand-labels --total-steps 300000 --eval-step 1000 --randaug 2 16 --batch-size 128 --teacher_lr 0.05 --student_lr 0.05 --weight-decay 5e-4 --ema 0.995 --nesterov --mu 7 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 8 --warmup-steps 5000 --uda-steps 5000 --student-wait-steps 3000 --teacher-dropout 0.2 --student-dropout 0.2 --amp
```

Monitoring training progress
```
tensorboard --logdir results
```

## Requirements
- python 3.6+
- torch 1.7+
- torchvision 0.8+
- tensorboard
- numpy
- tqdm
