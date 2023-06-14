# Practical Work in AI Master
This is a repository for the practical work in AI Master in SS2023 at the JKU University for the Institute for Computational Perception

1.) CP Resnet (Receptive Field Regularization-CNN) is trained on MNIST data (0-9 digits) 

2.) Then the model complexity is increased by increasing the width of the channels, and again it is trained on the MNIST dataset.

3.) Then this model is pruned by '''Torch Pruning''' (https://github.com/VainF/Torch-Pruning/tree/master) to get same complexity as the model in 1.)

4.) This pruned model is then fine-tuned to achieve at least better accuracy than 1.)


5.) The whole steps are repeated for ASC dataset (https://dcase.community/challenge2023/task-low-complexity-acoustic-scene-classification) instead of MNISt dataset.
