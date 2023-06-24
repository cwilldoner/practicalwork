# Practical Work in AI Master
This is a repository for the practical work in AI Master in SS2023 at the JKU University for the Institute for Computational Perception
A CNN for multi-class classification is trained on MNIST and ASC (https://dcase.community/challenge2023/task-low-complexity-acoustic-scene-classification) datasets and pruned by structured pruning technique. Those models are compared and the goal was to have better accuracy by lower model complexity due to structured pruning.

# Pruning technique #
The used pruning technique falls under structured pruning. The used pruning framework is from **Torch Pruning** (https://github.com/VainF/Torch-Pruning/tree/master) which is a Repo which consists of numerous pruning methods and functions with PyTorch. Those methods rely on Dependency Graphs. Those graphs are automatically created out of a neural network to group dependent units within a network, which serve as minimal removeable units, avoiding to destroy the overall network architecture and integrity. The framework serves several different high-level pruner methods which means the user does not have to dive into the dependency graph algorithm, but can use it in an more or less easy way. I opted for the **Magnitude Pruner**, since there was an example in their tutorial and it looked doable. The Magnitude Pruner removes weights with small magnitude in the network, resulting in a smaller and faster model without too much performance lost in accuracy. The user can define which importance to use i.e. which criterion should be used to remove filters from the network, which group reduction method e.g. mean, max, gaussian,... should be used, which norm should be used, the amount of channel sparsity and in how many iterations the channel sparsity should be reached. So those are still numerous parameters to set, where i sticked to the default ones (for pruning on MNIST) except for the channel sparsity and number of iterations (for pruning on ASC). The most important fact is to not prune the final classification layer. The paper of the Magnitude Pruner can be found here: https://arxiv.org/pdf/1608.08710.pdf
Other available high-level pruners are **BatchNormScalePruner** and **GroupNormPruner**

## MNIST
1. (small = original) CP Resnet (Receptive Field Regularization-CNN) is trained on **MNIST** data (0-9 digits) 

2. Then the model complexity is increased by increasing the width of the channels, and again it is trained on the MNIST dataset.

3. Then this model is structure-pruned to get same complexity as the model in 1.), specifically the **Magnitude Pruner** is used. The target pruning size should be equal or less than the small CP Resnet model. This pruned model is then fine-tuned to achieve at least better accuracy than 1.)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
## ASC
4. The whole steps are repeated for **ASC** dataset (https://dcase.community/challenge2023/task-low-complexity-acoustic-scene-classification) instead of MNISt dataset.



--------------------------------------------------------------------------------------------------------------------------------------------------------------------
## reproduce workflow for MNIST:

**_1. Train (small = original) CP Resnet (channel_width='24 48 72') on MNIST:_**

```
python ex_dcase.py --batch_size=256 --base_channels=32 --weight_decay=0.003 --lr=0.001 --n_epochs=50 --experiment_name="cpresnet_mnist_small" --mnist=1
```
This model has 59k params

**wandb Results:**

https://api.wandb.ai/links/dcase2023/sxurf83w

**_2. Train bigger model (and rename experiment_name):_**

```
python ex_dcase.py --batch_size=256 --base_channels=32 --weight_decay=0.003 --lr=0.001 --n_epochs=50 --experiment_name="cpresnet_mnist_big" --mnist=1 --channel_width='32 64 128'
```

This model 131316 params

**wandb Results:**

https://api.wandb.ai/links/dcase2023/spadhdku

**_3. Prune and fine-tune big model_**

```
python inference.py --batch_size=256 --base_channels=128 --weight_decay=0.003 --lr=0.001 --experiment_name="mnist_prune" --modelpath=trained_models/cpresnet_mnist_big_epoch=42-val_loss=0.04.ckpt --channel_width='32 64 128' --prune=1 --mnist=1
```

**Pruned model parameters (with 40% channel sparsity): 47349**

Fine-tuned iteratively on each prune stage

Run test on pruned model:

```
python inference.py --batch_size=256 --base_channels=128 --weight_decay=0.003 --lr=0.001 --experiment_name="mnist_prune" --modelpath=trained_models/pruned_mnist_prune.pth --channel_width='32 64 128' --prune=0 --mnist=1
```

**wandb Results:**

https://api.wandb.ai/links/dcase2023/f98vr3de

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
## reproduce workflow for ASC:

**_4. Train (small = original) CP Resnet (channel_width='24 48 72') on ASC:_**

```
python ex_dcase.py --batch_size=256 --base_channels=32 --weight_decay=0.003 --lr=0.001 --n_epochs=50 --experiment_name="cpresnet_asc_small" --mnist=0
```

**wandb Results:**

https://api.wandb.ai/links/dcase2023/vct47kfo

**_5. Train bigger model (and rename experiment_name):_**

```
python ex_dcase.py --batch_size=256 --base_channels=32 --weight_decay=0.003 --lr=0.001 --n_epochs=50 --experiment_name="cpresnet_asc_big" --mnist=0 --channel_width='32 64 128'
```

This model 131316 params

**wandb Results:**

https://api.wandb.ai/links/dcase2023/tjgwglew

**_6. Prune and fine-tune big model_**

```
python inference.py --batch_size=256 --base_channels=128 --weight_decay=0.003 --lr=0.001 --experiment_name="asc_prune" --modelpath=trained_models/cpresnet_asc_big_epoch=XX-val_loss=X.XX.ckpt --channel_width='32 64 128' --prune=1 --mnist=0
```

**Pruned model parameters (with 35% channel sparsity): 54706**

Fine-tuned iteratively on each prune stage (only 1 iteration stage used), same hyper params as ever

Run test on pruned model:

```
python inference.py --batch_size=256 --base_channels=128 --weight_decay=0.003 --lr=0.001 --experiment_name="asc_prune" --modelpath=trained_models/pruned_asc_prune.pth --channel_width='32 64 128' --prune=0 --mnist=0
```

**wandb Results:**

https://api.wandb.ai/links/dcase2023/p9g9unz3

The results are a bit more bad than the original small CPResnet, thus different hyper params will now be used, namely weight_decay is set to 0.001 and batch_size is set to 64:

```
python inference.py --batch_size=64 --base_channels=128 --weight_decay=0.001 --lr=0.001 --experiment_name="asc_prune_35_wd_bs64" --modelpath=trained_models/cpresnet_asc_big_epoch=49-val_loss=1.39.ckpt --channel_width='32 64 128' --prune=1 --mnist=0
```
**wandb Results:**
https://api.wandb.ai/links/dcase2023/ri1c686m

**Macro accuracy**

The macro accuracy of the pruned model **asc_prune_35_wd_bs64** (35% channel sparsity, weight_decay=0.001, batchsize=64, 54706 params) in comparison of the original small CPResnet **cpresnet_asc_small** (weight_decay=0.003, batchsize=256, 59000 params) is now slightly better as can be seen in the diagram below. It is even better as the big CPResnet with 131316 params, which has 0.5026 accuracy (all fine-tuned models start at this accuracy since from this model the pruning and fine-tuning process starts).
The other models in the diagram are **asc_prune_35_wd_bs** (35% channel sparsity, weight_decay=0.001, batchsize=128, 54706 params) and **asc_prune_35** (35% channel sparsity, weight_decay=0.003, batchsize=256, 54706 params)

![alt text](https://github.com/cwilldoner/practicalwork/blob/main/mac.png?raw=true)
