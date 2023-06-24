# Practical Work in AI Master
This is a repository for the practical work in AI Master in SS2023 at the JKU University for the Institute for Computational Perception

## MNIST
1. (small = original) CP Resnet (Receptive Field Regularization-CNN) is trained on **MNIST** data (0-9 digits) 

2. Then the model complexity is increased by increasing the width of the channels, and again it is trained on the MNIST dataset.

3. Then this model is structure-pruned by **Torch Pruning** (https://github.com/VainF/Torch-Pruning/tree/master) to get same complexity as the model in 1.), specifically the **Magnitude Pruner** is used. This method **removes weights with small magnitude** in the network, resulting in a smaller and faster model without too much performance lost in accuracy. The target pruning size should be equal or less than the small CP Resnet model. This pruned model is then fine-tuned to achieve at least better accuracy than 1.)

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

**Pruned model parameters (with 40% channel sparsity): XXXXX**

Fine-tuned iteratively on each prune stage

Run test on pruned model:

```
python inference.py --batch_size=256 --base_channels=128 --weight_decay=0.003 --lr=0.001 --experiment_name="asc_prune" --modelpath=trained_models/pruned_asc_prune.pth --channel_width='32 64 128' --prune=0 --mnist=0
```

**wandb Results:**
