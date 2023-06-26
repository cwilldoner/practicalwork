# Practical Work in AI Master
This is a repository for the practical work in AI Master in SS2023 at the JKU University for the Institute for Computational Perception. The code is fully based on PyTorch.

A CNN for multi-class classification is trained on MNIST ([1]) and ASC ([2]) datasets and pruned by structured pruning technique. Those models are compared and the goal is to have better accuracy by lower model complexity due to structured pruning. The work is based on the MALACH 23 course where a CNN was already trained on ASC, thus meaningful hyperparams where already found, but no pruning technique was applied.

## Pruning
Pruning is the technique of optimizing the network by its size to decrease computation and parameter storage overhead, without the loss of performance. It belongs to a group of network compression methods like Quantization and Knowledge Distillation. With Pruning, less significant neurons have to be detected and their dependencies across the network has to be measured, to not decrease the performance of the trained model after pruning. In general, pruning can be done before, during and after training. [3]
### Unstructured pruning
The most straight-forward technique is unstructured pruning where weights are simply set to zero. This does not alter the complexity of the network in an architectural manner, which does unfortunately not lead to any acceleration in matrix computation, since multiplications with zeros (and the accumulations), so called sparse matrix computations, are still performed. The advantage is, it is easy to implement and there is no problem with filter shapes inside the network since they stay the same.
### Structured pruning
The more complex way is to remove whole filters from the network, since removing a filter results in removing the feature map it outputs too and the consecutive kernels from the consecutive layer. [3]

The used pruning framework is **Torch Pruning** ([4]) which consists of numerous pruning methods and functions based on PyTorch, but not using the built-in library torch.nn.utils.prune which is "just" using a zero mask. The mentioned framework is all about structured pruning methods and they rely on Dependency Graphs. Those graphs are automatically created out of a neural network to group dependent units within a network, which serve as minimal removeable units, avoiding to destroy the overall network architecture and integrity. The framework serves several different high-level pruner methods which means the user does not have to dive into the dependency graph algorithm, but can use it in an more or less easy way. I opted for the **Magnitude Pruner**, since there was an example in their tutorial and it looked doable. The Magnitude Pruner removes weights with small magnitude in the network, resulting in a smaller and faster model without too much performance lost in accuracy. The user can define which importance to use i.e. which criterion should be used to remove filters from the network, which group reduction method e.g. mean, max, gaussian,... should be used, which norm should be used, the amount of channel sparsity and in how many iterations the channel sparsity should be reached. So those are still numerous parameters to set, where i sticked to the default ones (for pruning on MNIST) except for the channel sparsity and number of iterations (for pruning on ASC). The most important fact is to not prune the final classification layer. The paper of the Magnitude Pruner can be found here: [5]

Other available high-level pruners are **BatchNormScalePruner** and **GroupNormPruner**

## Workflow

At first, the pipeline is set up with a minimal example, in this case the MNIST dataset is used. Here, no pre-processing of the images is used.

1. (small = original) CP Resnet (Receptive Field Regularization-CNN) is trained on **MNIST** data (0-9 digits) 

2. Then the model complexity is increased by increasing the width of the channels, and again it is trained on the MNIST dataset.

3. Then this model is structure-pruned to get same complexity as the model in 1.), specifically the **Magnitude Pruner** is used. The target pruning size should be equal or less than the small CP Resnet model. This pruned model is then fine-tuned to achieve at least better accuracy than 1.)

4. The whole steps are repeated for **ASC** dataset.

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

The channel sparsity was set to **35%** to achieve approximately the same amount of params than the small CPResnet model. It resulted in even about 5k less params of **54706**.
Best results were found when only one iteration stage was used. The hyper params where the same as before.

**wandb Results:**

https://api.wandb.ai/links/dcase2023/p9g9unz3

The results on accuracy were a bit worse than the original small CPResnet, thus different hyper params will now be used, namely **weight_decay** is set to 0.001 and **batch_size** is set to 64:

```
python inference.py --batch_size=64 --base_channels=128 --weight_decay=0.001 --lr=0.001 --experiment_name="asc_prune_35_wd_bs64" --modelpath=trained_models/cpresnet_asc_big_epoch=XX-val_loss=X.XX.ckpt --channel_width='32 64 128' --prune=1 --mnist=0
```
**wandb Results:**
https://api.wandb.ai/links/dcase2023/ri1c686m

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Results
### Magnitude Pruner with loading pre-trained CPResnet
The macro accuracy of the pruned model **asc_prune_35_wd_bs64** (35% channel sparsity, weight_decay=0.001, batchsize=64, 54706 params) in comparison of the original small CPResnet **cpresnet_asc_small** (weight_decay=0.003, batchsize=256, 59000 params) is now slightly better as can be seen in the diagram below. It is even better as the big CPResnet with 131316 params, which has 0.5026 accuracy (all fine-tuned models start at this accuracy since from this model the pruning and fine-tuning process starts).
The other models in the diagram are **asc_prune_35_wd_bs** (35% channel sparsity, weight_decay=0.001, batchsize=128, 54706 params) and **asc_prune_35** (35% channel sparsity, weight_decay=0.003, batchsize=256, 54706 params)

![alt text](https://github.com/cwilldoner/practicalwork/blob/main/mac.png?raw=true)

| Model  | Parameter | Accuracy |
| ------------- | ------------- | ------------- |
| cpresnet_asc_small  | 59000  | 0.4966  |
| cpresnet_asc_big  | 131316  | 0.5026  |
| **asc_prune_35_wd_bs64**  | **54706**  | **0.505**  |
| asc_prune_35_wd_bs  | 54706  | 0.4975  |
| asc_prune_35  | 54706  | 0.4866  |

### Other pruners
When using the other available pruners like BatchNormScalePruner(_bn) and GroupNormPruner (_gn) the results vary marginally, but BatchNormScalePruner seems to perform best all of three pruner types.
What is more interesting is that you do not need to train the CPResnet before pruning. You just have to prune first your "empty" network, and after that (or during if you use more iteration steps for pruning) you train the network. The results seem even to be better without using a pre-trained CPResnet (=loading the model from checkpoint). In the diagram below the _fromscratch models are the ones with pruning before training (so far only Magnitude Pruner (**asc_prune_35_wd_bs64_fromscratch**) and BatchNormScalePruner (**asc_prune_35_wd_bs64_fromscratch_bn**) was used from scratch).
![alt text](https://github.com/cwilldoner/practicalwork/blob/main/mac3.png?raw=true)

| Model  | Parameter | Accuracy |
| ------------- | ------------- | ------------- |
| **asc_prune_35_wd_bs64_fromscratch**  | **54706**  | **0.5353**  |
| asc_prune_35_wd_bs64_fromscratch_bn  | 54706  | 0.5275  |

In general one can say pruning does already meaningful optimization of the network without significant loss of performance.

## References
[1] https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html

[2] https://dcase.community/challenge2023/task-low-complexity-acoustic-scene-classification

[3] https://towardsdatascience.com/neural-network-pruning-101-af816aaea61

[4] https://github.com/VainF/Torch-Pruning/tree/master

[5] https://arxiv.org/pdf/1608.08710.pdf
