# Practical Work in AI Master
This is a repository for the practical work in AI Master in SS2023 at the JKU University for the Institute for Computational Perception

### MNIST
1.) (small = original) CP Resnet (Receptive Field Regularization-CNN) is trained on MNIST data (0-9 digits) 

2.) Then the model complexity is increased by increasing the width of the channels, and again it is trained on the MNIST dataset.

3.) Then this model is pruned by '''Torch Pruning''' (https://github.com/VainF/Torch-Pruning/tree/master) to get same complexity as the model in 1.)

4.) This pruned model is then fine-tuned to achieve at least better accuracy than 1.)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
### ASC
5.) The whole steps are repeated for ASC dataset (https://dcase.community/challenge2023/task-low-complexity-acoustic-scene-classification) instead of MNISt dataset.



## Workflow:

1.) Train (small = original) CP Resnet (channel_width='24 48 72') on MNIST:

python ex_dcase.py --batch_size=256 --base_channels=32 --weight_decay=0.003 --lr=0.001 --n_epochs=50 --experiment_name="cpresnet_mnist_small" --mnist=1

This model has 59k params
wandb Results:

https://api.wandb.ai/links/dcase2023/sxurf83w

2.) Train bigger model (and rename experiment_name):

python ex_dcase.py --batch_size=256 --base_channels=32 --weight_decay=0.003 --lr=0.001 --n_epochs=50 --experiment_name="cpresnet_mnist_big" --mnist=1 --channel_width='32 64 128'

This model 131k params
wandb Results:

https://api.wandb.ai/links/dcase2023/spadhdku

3.) Prune big model
