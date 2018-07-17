# Mixture of experts layers for Keras

This repository contains Keras layers implementing convolutional and dense mixture of experts models.

## Dense mixture of experts layer

The file `DenseMoE.py` contains a Keras layer implementing a dense mixture of experts model:

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}&space;=&space;\sum_{k=1}^K&space;g(\mathbf{v}_k^\top&space;\mathbf{x}&space;&plus;&space;b_k)f(\mathbf{W}_k\mathbf{x}&space;&plus;&space;\mathbf{c}_k)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}&space;=&space;\sum_{k=1}^K&space;g(\mathbf{v}_k^\top&space;\mathbf{x}&space;&plus;&space;b_k)f(\mathbf{W}_k\mathbf{x}&space;&plus;&space;\mathbf{c}_k)" title="\mathbf{y} = \sum_{k=1}^K g(\mathbf{v}_k^\top \mathbf{x} + b_k)f(\mathbf{W}_k\mathbf{x} + \mathbf{c}_k)" /></a>

This layer can be used in the same way as a `Dense` layer. Some of its main arguments are as follows:
* `units`: the output dimensionality
* `n_experts`: the number of experts (<a href="https://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?K" title="K" /></a>)
* `expert_activation`: activation function for the expert model (<a href="https://www.codecogs.com/eqnedit.php?latex=f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f" title="f" /></a>)
* `gating_activation`: activation function for the gating model (<a href="https://www.codecogs.com/eqnedit.php?latex=g" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g" title="g" /></a>)

Please see `DenseMoE.py` for additional arguments. 

## Convolutional mixture of experts layer

The file `ConvolutionalMoE.py` contains Keras layers implementing 1D, 2D, and 3D convolutional mixture of experts models:

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}&space;=&space;\sum_{k=1}^K&space;g(\mathbf{v}_k^\top&space;\mathbf{x}&space;&plus;&space;b_k)f(\mathbf{W}_k&space;*&space;\mathbf{x}&space;&plus;&space;\mathbf{c}_k)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}&space;=&space;\sum_{k=1}^K&space;g(\mathbf{v}_k^\top&space;\mathbf{x}&space;&plus;&space;b_k)f(\mathbf{W}_k&space;*&space;\mathbf{x}&space;&plus;&space;\mathbf{c}_k)" title="\mathbf{y} = \sum_{k=1}^K g(\mathbf{v}_k^\top \mathbf{x} + b_k)f(\mathbf{W}_k * \mathbf{x} + \mathbf{c}_k)" /></a>

where `*` denotes a convolution operation. These layers can be used in the same way as the corresponding standard convolutional layers (`Conv1D`, `Conv2D`, `Conv3D`).

## Examples

The file `conv_moe_demo.py` contains an example demonstrating how to use these layers. The example here is based on the `cifar10_cnn.py` file in the [keras/examples](https://github.com/keras-team/keras/tree/master/examples) folder and it builds a simple convolutional deep network, using either standard convolutional and dense layers or the corresponding mixture of experts layers, and compares the performance of the two models on CIFAR-10 image recognition benchmark. 

The figure below compares the validation accuracy of the standard convolutional model with that of the mixture of experts model with two experts (K=2, I have observed signs of overfitting for larger numbers of experts). The error bars are standard errors over 4 independent repetitions. The mixture of experts model performs significantly better than the standard convolutional model, although admittedly this is not a controlled experiment as the mixture of experts model has more parameters (please let me know if you would be interested in helping me run more controlled experiments comparing the performance of these two models).

![](https://github.com/eminorhan/mixture-of-experts/blob/master/moe_vs_cnn.png)

The file `jacobian_moe_demo.py` contains another example illustrating how to use the dense mixture of experts layer. The example here essentially implements the simulations reported in [this blog post](https://severelytheoretical.wordpress.com/2018/06/08/the-softmax-bottleneck-is-a-special-case-of-a-more-general-phenomenon/).

I have tested these examples with Tensorflow v1.8.0 and Keras v2.0.9 on my laptop without using a GPU. Other configurations may or may not work. Please let me know if you have any trouble running these examples. 
