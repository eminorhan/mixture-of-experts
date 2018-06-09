# Mixture of experts layer for Keras

This repository contains a Keras layer implementing a dense mixture of experts model:

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}&space;=&space;\sum_{k=1}^K&space;g(\mathbf{v}_k^\top&space;\mathbf{x}&space;&plus;&space;b_k)f(\mathbf{W}_k\mathbf{x}&space;&plus;&space;\mathbf{c}_k)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}&space;=&space;\sum_{k=1}^K&space;g(\mathbf{v}_k^\top&space;\mathbf{x}&space;&plus;&space;b_k)f(\mathbf{W}_k\mathbf{x}&space;&plus;&space;\mathbf{c}_k)" title="\mathbf{y} = \sum_{k=1}^K g(\mathbf{v}_k^\top \mathbf{x} + b_k)f(\mathbf{W}_k\mathbf{x} + \mathbf{c}_k)" /></a>

Some of the main arguments are as follows:
* `units`: the output dimensionality
* `n_experts`: the number of experts (<a href="https://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?K" title="K" /></a>)
* `expert_activation`: activation function for the expert model (<a href="https://www.codecogs.com/eqnedit.php?latex=f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f" title="f" /></a>)
* `gating_activation`: activation function for the gating model (<a href="https://www.codecogs.com/eqnedit.php?latex=g" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g" title="g" /></a>)

Please see `MixtureOfExperts.py` for additional arguments. The file `moe_demo.py` contains an example demonstrating how to use this layer. The example there essentially implements the simulations reported in [this blog post](https://severelytheoretical.wordpress.com/2018/06/08/the-softmax-bottleneck-is-a-special-case-of-a-more-general-phenomenon/).
