# Practical Work in AI Master
This is a repository for the practical work in AI Master in SS2023 at the JKU University for the Institute for Computational Perception. The code is fully based on PyTorch. The course is a continuation of the MALACH23 course from the Institute for Computational Perception.

A CNN for multi-class classification is trained, first on MNIST [1] dataset to set up the pipeline. After that, the actual aim is to train the network on ASC [2] dataset and prune it by structured pruning technique. The MNIST dataset is chosen because you do not need any pre-processing steps of the input images and it has 10 classes, like the ASC dataset. 

The used CNN network is a Receptive Field Regularization CNN [6], originating from the Institute of Computational Perception, afterwards in this documentation it is called CPResnet. Further, it is differentiated between a CPResnet original network and the CPResnet pruned network. The CPResnet original has a defined number of parameters, which has to be underbid by structured pruned version CPResnet pruned. Further, the accuracy and loss of the CPResnet pruned should be better than the CPResnet original network. 

## Pruning
Pruning is the technique of optimizing the network by its size to decrease computation and parameter storage overhead, without the loss of performance. It belongs to a group of network compression methods like Quantization and Knowledge Distillation. With Pruning, less significant neurons have to be detected and their dependencies across the network has to be measured, to not decrease the performance of the trained model after pruning. In general, pruning can be done before, during and after training. [3]
### Unstructured pruning
The most straight-forward technique is unstructured pruning where weights are simply set to zero. This does not alter the complexity of the network in an architectural manner, which does unfortunately not lead to any acceleration in matrix computation, since multiplications with zeros (and the accumulations), so called sparse matrix computations, are still performed. The advantage is, it is easy to implement and there is no problem with filter shapes inside the network since they stay the same.
### Structured pruning
The more complex way is to remove whole filters from the network, since removing a filter results in removing the feature map it outputs too and the consecutive kernels from the consecutive layer. [3]

The used pruning framework is **Torch Pruning** [4] which consists of numerous pruning methods and functions based on PyTorch, but not using the built-in library torch.nn.utils.prune which is "just" using a zero mask. The mentioned framework is all about structured pruning methods and they rely on Dependency Graphs. Those graphs are automatically created out of a neural network to group dependent units within a network, which serve as minimal removeable units, avoiding to destroy the overall network architecture and integrity. The framework serves several different high-level pruner methods which means the user does not have to dive into the dependency graph algorithm, but can use it in an more or less easy way. 

In this work two high-level pruners where mainly experimented with, the **Magnitude Pruner** and the **BatchNormScalePruner**. 
Since there was an example for the Magnitude Pruner in their tutorial, i used this method first. For all types of pruners, the user can define which importance to use i.e. which criterion should be used to remove filters from the network, which group reduction method e.g. mean, max, gaussian,... should be used, which norm should be used, the amount of channel sparsity and in how many iterations the channel sparsity should be reached. So those are still numerous parameters to set, where i sticked to the default ones (for pruning on MNIST) except for the channel sparsity and number of iterations (for pruning on ASC). The most important fact is to not prune the final classification layer.
#### Magnitude Pruner [5]
The Magnitude Pruner removes weights with small magnitude in the network, resulting in a smaller and faster model without too much performance loss in accuracy. The paper of the Magnitude Pruner can be found here:
![alt text](https://github.com/cwilldoner/practicalwork/blob/main/mag_prune1.png?raw=true)

$`\textbf{x}_i`$ is a feature map, consisting of $`n_i`$ feature maps, each of size $`h_i`$ and $`w_i`$. E.g. for the two consecutive layers ```self.conv1 = Conv2d(1,6,(5,5))``` and ```self.conv2 = Conv2d(6,16,(5,5))```, $`n_1=1`$, $`n_2=6`$ and $`n_3=16`$. A kernel matrix $`F_i \in \mathbb{R}^{n_i \times n_{i+1} \times k \times k}`$ consists of $`n_{i+1}`$ kernel filters e.g. for the first kernel matrix in the above figure, there would be 6 kernel filters, each of size ```1x(5,5)```, because a kernel filter $`F_{i,j} \in \mathbb{R}^{n_i \times k \times k}`$ consists of $`n_i`$ kernels (thus ```1x(5,5)```) of size $`\kappa \in \mathbb{R}^{k \times k}`$. The number of operations for a whole layer without pruning is $`n_{i+1}*(n_i*k*k*h_{i+1}*w_{i+1})`$. When a filter is removed, the number of operations is reduced by $`(n_i*k*k*h_{i+1}*w_{i+1})`$, since a whole feature map is removed. The now removed feature map leads to an removed filter in the next layer which leads to an additional reduction of $`(n_{i+2}*k*k*h_{i+2}*w_{i+2})`$ operations. With a hyperparameter $m$ one can define how many filters should be removed from the kernel matrix (number of blue columns in the above figure).

The weights of a filter in each layer are a measure of importance i.e. low weights mean low importance and vice versa. The relative importance is the importance of each filter to the sum of its absolute weights (when using L1 norm) from the whole layer.
The procedure of pruning $m$ filters from the $`i`$th convolutional layer for L1 norm is as follows:
1. For each filter $`F_{i,j}`$ , calculate the sum of its absolute kernel weights $`s_j = \sum \sum |\kappa_l|`$ e.g. for layer 1, i.e. $`i=1`$ and $`i+1=2`$ -> $`n_2=6`$: $`s_1 = (|(5 \times 5)|) = 0.6`$, $`s_2 = (|(5 \times 5)) = 0.1 `$, ..., $`s_6 = (|(5 \times 5)) = -1.3`$
2. Sort the filters by $`s_j`$, e.g. $`s_2 > s_1 > s_3 > s_4 > s_5 > s_6`$
3. Prune $`m`$ filters with the smallest sum values and their corresponding feature maps, e.g. $`m=4`$, thus $`s_6`$, $`s_5`$, $`s_4`$, $`s_3`$ will be removed. This means the dimension of the kernel matrix across the $`n+1`$ dimension will be decreased, thus the channel amount will be decreased (e.g. for layer 1 -> $`i=1`$ -> ```self.conv1 = Conv2d(1,2, (5,5))```). The
kernels in the next convolutional layer corresponding to the pruned feature maps are also
removed, thus the dimension of the kernel matrix across the $`n`$ dimension will be decreased, thus the channel amount will be decreased (e.g. for layer 2 -> $`i+1=2`$ for $`i=1`$ -> ```self.conv2 = Conv2d(2,16,(5,5))```).
4. A new kernel matrix is created for both the $`i`$th and $`i`$ + 1th layers, and the remaining kernel
weights are copied to the new model.

This is done for each layer in the network (except final classification layer).

#### BatchNormalizationScale Pruner [7]
The BatchNormalizationScale Pruner focuses on the scaling factor $`\gamma`$ from a Batch Normalization layer ([PyTorch BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)). This parameter scales the output distribution of each channel.
![alt text](https://github.com/cwilldoner/practicalwork/blob/main/bn_prune1.png?raw=true)

In this approach the L1 norm is not applied on the weights, but on the channel-wise batch norm scaling factor.

## Baseline CPResnet original
In the MALACH23 course, the **CPResnet original** network was configured to have 47028 parameters to achieve satisfying results. This number of parameters is the upper limit for the **CPResnet pruned** network.
To train the CPResnet original, several hyperparameters where found in the preceding course MALACH23. The most important used hyperparams for training are listed in the table below:
| model  | CPResnet original | 
| ------------- | ------------- |
| **parameters**  | **47028**  |
| batch size  | 256  |
| channels_multiplier | 1 |
| base channels  | 32  |
| weight decay  | 0.003  |
| learning rate  | 0.001  |
| epochs  | 50  |
| experiment name  | cpresnet_asc_small  |
| mnist  | 0  |
| learning rate scheduler | lambdaLR |

To start training type
```
python ex_dcase.py --batch_size=256 --base_channels=32 --channels_multiplier=1 --weight_decay=0.003 --lr=0.001 --n_epochs=50 --experiment_name="cpresnet_asc_small" --mnist=0
```

Training results:
![alt text](https://github.com/cwilldoner/practicalwork/blob/main/asc_small.png?raw=true)

This is the baseline. This experiment is executed three times and the average of the test accuracy (mac_acc) and test loss (test_loss) on the test dataset for those experiments is taken and shown in the next table below:
| model  | CPResnet original | 
| ------------- | ------------- |
| average accuracy  | 0.50553  |
| average loss  | 1.38185  |

## Increase size of CPResnet original
Now, the size of the CPResnet original network is increased to get a bigger model, which can be pruned, to show the difference of the original to a pruned version.
The size is increased by setting the parameter **channel multiplier** to 2. This increases the parameters from 47028 parameters to 131316 parameters.
This new big model, in the following called **CPResnet big**, is trained with the same hyperparameters, except for the channel multiplier of course.
| model  | CPResnet big | 
| ------------- | ------------- |
| **parameters**  | **131316**  |
| batch size  | 256  |
| channels_multiplier | 2 |
| base channels  | 32  |
| weight decay  | 0.003  |
| learning rate  | 0.001  |
| epochs  | 50  |
| experiment name  | cpresnet_asc_big  |
| mnist  | 0  |
| learning rate scheduler | lambdaLR |

To start training type
```
python ex_dcase.py --batch_size=256 --base_channels=32 --channels_multiplier=2 --weight_decay=0.003 --lr=0.001 --n_epochs=50 --experiment_name="cpresnet_asc_big" --mnist=0
```

| model  | CPResnet big | 
| ------------- | ------------- |
| average accuracy  | 0.50623  |
| average loss  | 1.38185  |

This is an intermediate result for the whole pruning experiment, thus there is no need to average the results from multiple runs, since we can only take one model to proceed. 

## Experiments with pruning
Experiments were conducted with different types of learning rate scheduler, training from scratch or not, different starting rates for learning rate and weight decay, number of epochs, types of pruners.
### Prune CPResnet big
In a first step, the pre-trained CPResnet big is pruned by different pruner methods supported by the Torch Pruner framework. The pruners work the same in the way, that a pre-trained model is loaded and in customized number of iterations the network is pruned, and in the same time fine-tuned.
In this step, one has to use the **inference.py** script instead of the ex_dcase.py script. Here the hyperparameters stand for the fine-tuning, not for the training from scratch (further details on this in the next sections). The most important paramters are listed in the table below: 
| model  | CPResnet pruned | 
| ------------- | ------------- |
| **parameters after pruning**  | **44784**  |
| batch size  | 256  |
| channels_multiplier | 2 |
| base channels  | 32  |
| weight decay  | 0.001  |
| learning rate  | 0.0001  |
| epochs  | 50  |
| experiment name  | cpresnet_asc_pruned  |
| mnist  | 0  |
| pruned | 1 |
| channel sparsity | 0.41 |
| learning rate scheduler | lambdaLR |

The parameter **channel sparsity** is important to regulate the number of parameters. We want to have approximately the same as **CPResnet original** (47028) to be comparable. Thus the parameter resulted in to remove 41% of the channels of the whole network. This resulted in 44784 parameters, thus slightly less than the original. This parameter is set once in the code, so it is not necessary to make a hyperparameter of it. The pre-trained CPResnet big is stored under the trained_models folder and the .ckpt file with the best validation loss should be loaded.

To start pruning type
```
python inference.py --batch_size=256 --base_channels=32 --weight_decay=0.001 --lr=0.0001 --experiment_name="cpresnet_asc_pruned" --modelpath=trained_models/cpresnet_asc_big_epoch=XX-val_loss=X.XX.ckpt --prune=1 --mnist=0
```

By default, the Magnitude Pruner (mag) is used, but with parameter ```--pruner``` you can define bn (Batch Normalization Scale Pruner) and gn (GroupNorm Pruner). With parameter ```--iterative_steps``` you can also define in how many steps the pruning process should happen to reach the 41% sparsity, but by default 1 worked the best.

The results seemed not too bad, but with the chosen parameters it seemed the results will not get better. It was trained with 50 epochs and 80 epochs and it seemed it was stuck with the learning rate scheduler.
![alt text](https://github.com/cwilldoner/practicalwork/blob/main/pretrained_results.png?raw=true)
So in a next experiment a different learning rate scheduler was used. With the different learning rate schedulers from PyTorch [How to adjust learning rate](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) one might achieve better results. I experimented with the ReduceLROnPlateau scheduler which decreases learning rate when a metric is stuck improving for a defined number of epochs.

REMARK:

You need to load the model as .pth (not as .ckpt) when changing the learning rate scheduler. The .pth lies in the same folder as the .ckpt file.

To start pruning type with ReduceLROnPlateau scheduler type
```
python inference.py --batch_size=256 --base_channels=32 --weight_decay=0.0001 --lr=0.0001 --experiment_name="cpresnet_asc_pruned_pretrained_mag_redLR_1" --modelpath=trained_models/cpresnet_asc_big.pth --pruner='mag' --prune=1 --mnist=0 --iterative_steps=1 --scheduler="reduceLR" --n_epochs=80
```

Result for the best fine-tuned experiment is shown in the picture below:
![alt text](https://github.com/cwilldoner/practicalwork/blob/main/pretrained_l1.png?raw=true)


### Pruning an empty network (from scratch)
In a last experiment, it was found that it need not to be a pre-trained model to be loaded for the pruners. One can take an untrained **CPResnet big** i.e. just initialize the network module with the untrained class SimpleDCASELitModule ```pl_module = SimpleDCASELitModule(config)```, and feed it into the pruner. The fine-tuning process now is the actual training process, but it happens during the pruning iterations. 
| model  | CPResnet pruned from scratch | 
| ------------- | ------------- |
| **parameters after pruning**  | **44784**  |
| batch size  | 256  |
| channels_multiplier | 2 |
| base channels  | 32  |
| weight decay  | 0.001  |
| learning rate  | 0.0001  |
| epochs  | 150  |
| experiment name  | cpresnet_asc_pruned_fs  |
| mnist  | 0  |
| pruned | 1 |
| channel sparsity | 0.41 |
| learning rate scheduler | reduce LR on plateau |
| pruner method | magnitude |

To start training and pruning from an empty network type
```
python inference.py --batch_size=256 --base_channels=32 --weight_decay=0.0001 --lr=0.0001 --experiment_name="cpresnet_asc_pruned_fs_4_mag_LRplat" --pruner='mag' --prune=1 --mnist=0 --iterative_steps=1 --n_epochs=150 --from_scratch=1 --scheduler="reduceLR"
```



Result for the best from-scratch experiment is shown in the picture below:
![alt text](https://github.com/cwilldoner/practicalwork/blob/main/best_results.png?raw=true)


This experiment is also executed several times and the average of the accuracy and loss for those experiments is taken and shown in the next table below:
| model  | CPResnet pruned | 
| ------------- | ------------- |
| average accuracy  | 0.5083  |
| average loss  | 1.383  |


## Conclusion
In several experiments it was shown that structured pruning is a way of minimizing a model size by maintaining the accuracy. It was experimented with training with more epochs, different learning rate schedulers, weight decay and learning rate, and different pruner methods (Magnitude and BatchNorm Scale). The fine-tuning after pruning is absolutely necessary to maintain the accuracy, and while good results were achieved with less model parameters than the original CPResnet, all experiments differed slightly, and it was hard to increase the accuracy after pruning, in this work it was not possible to make a huge step to go beyond the original results.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Appendix: Workflow for MNIST

**_1. Train (small = original) CP Resnet on MNIST:_**

```
python ex_dcase.py --batch_size=256 --base_channels=32 --weight_decay=0.003 --lr=0.001 --n_epochs=50 --experiment_name="cpresnet_mnist_small" --mnist=1
```

**wandb Results:**

https://api.wandb.ai/links/dcase2023/sxurf83w

**_2. Train bigger model:_**

```
python ex_dcase.py --batch_size=256 --base_channels=32 --weight_decay=0.003 --lr=0.001 --n_epochs=50 --experiment_name="cpresnet_mnist_big" --mnist=1 --channels_multiplier=2
```

**wandb Results:**

https://api.wandb.ai/links/dcase2023/spadhdku

**_3. Prune and fine-tune big model_**

```
python inference.py --batch_size=256 --base_channels=128 --weight_decay=0.003 --lr=0.001 --experiment_name="mnist_prune" --modelpath=trained_models/cpresnet_mnist_big_epoch=42-val_loss=0.04.ckpt --channels_multiplier=2 --prune=1 --mnist=1
```

**Pruned model parameters (with 41% channel sparsity):**

Fine-tuned iteratively on each prune stage

Run test on pruned model:

```
python inference.py --batch_size=256 --base_channels=128 --weight_decay=0.003 --lr=0.001 --experiment_name="mnist_prune" --modelpath=trained_models/pruned_mnist_prune.pth --channels_multiplier=2 --prune=0 --mnist=1
```

**wandb Results:**

https://api.wandb.ai/links/dcase2023/f98vr3de

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

## References
[1] https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html

[2] https://dcase.community/challenge2023/task-low-complexity-acoustic-scene-classification

[3] https://towardsdatascience.com/neural-network-pruning-101-af816aaea61

[4] https://github.com/VainF/Torch-Pruning/tree/master

[5] H. Li, A. Kadav, Ig. Durdanovic, H. Samet, H. P. Graf (2017). Pruning Filters for Efficient ConvNets. ICLR 2017 5th International Conference on Learning Representations

[6] F. Schmid, S. Masoudian, K. Koutini, G. Widmer (2022). CP-JKU Submission to DCASE22: Distilling Knowledge for Low-Complexity
Convolutional Neural Networks from a Patchout Audio Transformer

[7] Z. Liu1, J. Li, Z. Shen, G. Huang, S. Yan, C. Zhang (2017). Learning Efficient Convolutional Networks through Network Slimming. ICCV 2017
International Conference on Computer Vision
