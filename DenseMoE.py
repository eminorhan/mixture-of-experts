#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:43:34 2018 @author: Emin Orhan
"""
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.initializers import RandomUniform
from keras.engine.topology import Layer, InputSpec

class DenseMoE(Layer):
    """Mixture-of-experts layer.
    Implements: y = sum_{k=1}^K g(v_k * x) f(W_k * x)
        # Arguments
        units: Positive integer, dimensionality of the output space.
        n_experts: Positive integer, number of experts (K).
        expert_activation: Activation function for the expert model (f).
        gating_activation: Activation function for the gating model (g).
        use_expert_bias: Boolean, whether to use biases in the expert model.
        use_gating_bias: Boolean, whether to use biases in the gating model.
        expert_kernel_initializer_scale: Float, scale of Glorot uniform initialization for expert model weights.
        gating_kernel_initializer_scale: Float, scale of Glorot uniform initialization for gating model weights.
        expert_bias_initializer: Initializer for the expert biases.
        gating_bias_initializer: Initializer fot the gating biases.
        expert_kernel_regularizer: Regularizer for the expert model weights.
        gating_kernel_regularizer: Regularizer for the gating model weights.
        expert_bias_regularizer: Regularizer for the expert model biases.
        gating_bias_regularizer: Regularizer for the gating model biases.
        expert_kernel_constraint: Constraints for the expert model weights.
        gating_kernel_constraint: Constraints for the gating model weights.
        expert_bias_constraint: Constraints for the expert model biases.
        gating_bias_constraint: Constraints for the gating model biases.
        activity_regularizer: Activity regularizer.
    # Input shape
        nD tensor with shape: (batch_size, ..., input_dim).
        The most common situation would be a 2D input with shape (batch_size, input_dim).
    # Output shape
        nD tensor with shape: (batch_size, ..., units).
        For example, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).
    """
    def __init__(self, units,
                 n_experts,
                 expert_activation=None,
                 gating_activation=None,
                 use_expert_bias=True,
                 use_gating_bias=True,
                 expert_kernel_initializer_scale=1.0,
                 gating_kernel_initializer_scale=1.0,
                 expert_bias_initializer='zeros',
                 gating_bias_initializer='zeros',
                 expert_kernel_regularizer=None,
                 gating_kernel_regularizer=None,
                 expert_bias_regularizer=None,
                 gating_bias_regularizer=None,
                 expert_kernel_constraint=None,
                 gating_kernel_constraint=None,
                 expert_bias_constraint=None,
                 gating_bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseMoE, self).__init__(**kwargs)
        self.units = units
        self.n_experts = n_experts

        self.expert_activation = activations.get(expert_activation)
        self.gating_activation = activations.get(gating_activation)

        self.use_expert_bias = use_expert_bias
        self.use_gating_bias = use_gating_bias

        self.expert_kernel_initializer_scale = expert_kernel_initializer_scale
        self.gating_kernel_initializer_scale = gating_kernel_initializer_scale

        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gating_bias_initializer = initializers.get(gating_bias_initializer)

        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gating_kernel_regularizer = regularizers.get(gating_kernel_regularizer)

        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gating_bias_regularizer = regularizers.get(gating_bias_regularizer)

        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gating_kernel_constraint = constraints.get(gating_kernel_constraint)

        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gating_bias_constraint = constraints.get(gating_bias_constraint)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        expert_init_lim = np.sqrt(3.0*self.expert_kernel_initializer_scale / (max(1., float(input_dim + self.units) / 2)))
        gating_init_lim = np.sqrt(3.0*self.gating_kernel_initializer_scale / (max(1., float(input_dim + 1) / 2)))

        self.expert_kernel = self.add_weight(shape=(input_dim, self.units, self.n_experts),
                                      initializer=RandomUniform(minval=-expert_init_lim,maxval=expert_init_lim),
                                      name='expert_kernel',
                                      regularizer=self.expert_kernel_regularizer,
                                      constraint=self.expert_kernel_constraint)

        self.gating_kernel = self.add_weight(shape=(input_dim, self.n_experts),
                                      initializer=RandomUniform(minval=-gating_init_lim,maxval=gating_init_lim),
                                      name='gating_kernel',
                                      regularizer=self.gating_kernel_regularizer,
                                      constraint=self.gating_kernel_constraint)

        if self.use_expert_bias:
            self.expert_bias = self.add_weight(shape=(self.units, self.n_experts),
                                        initializer=self.expert_bias_initializer,
                                        name='expert_bias',
                                        regularizer=self.expert_bias_regularizer,
                                        constraint=self.expert_bias_constraint)
        else:
            self.expert_bias = None

        if self.use_gating_bias:
            self.gating_bias = self.add_weight(shape=(self.n_experts,),
                                        initializer=self.gating_bias_initializer,
                                        name='gating_bias',
                                        regularizer=self.gating_bias_regularizer,
                                        constraint=self.gating_bias_constraint)
        else:
            self.gating_bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):

        expert_outputs = tf.tensordot(inputs, self.expert_kernel, axes=1)
        if self.use_expert_bias:
            expert_outputs = K.bias_add(expert_outputs, self.expert_bias)
        if self.expert_activation is not None:
            expert_outputs = self.expert_activation(expert_outputs)

        gating_outputs = K.dot(inputs, self.gating_kernel)
        if self.use_gating_bias:
            gating_outputs = K.bias_add(gating_outputs, self.gating_bias)
        if self.gating_activation is not None:
            gating_outputs = self.gating_activation(gating_outputs)

        output = K.sum(expert_outputs * K.repeat_elements(K.expand_dims(gating_outputs, axis=1), self.units, axis=1), axis=2)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'n_experts':self.n_experts,
            'expert_activation': activations.serialize(self.expert_activation),
            'gating_activation': activations.serialize(self.gating_activation),
            'use_expert_bias': self.use_expert_bias,
            'use_gating_bias': self.use_gating_bias,
            'expert_kernel_initializer_scale': self.expert_kernel_initializer_scale,
            'gating_kernel_initializer_scale': self.gating_kernel_initializer_scale,
            'expert_bias_initializer': initializers.serialize(self.expert_bias_initializer),
            'gating_bias_initializer': initializers.serialize(self.gating_bias_initializer),
            'expert_kernel_regularizer': regularizers.serialize(self.expert_kernel_regularizer),
            'gating_kernel_regularizer': regularizers.serialize(self.gating_kernel_regularizer),
            'expert_bias_regularizer': regularizers.serialize(self.expert_bias_regularizer),
            'gating_bias_regularizer': regularizers.serialize(self.gating_bias_regularizer),
            'expert_kernel_constraint': constraints.serialize(self.expert_kernel_constraint),
            'gating_kernel_constraint': constraints.serialize(self.gating_kernel_constraint),
            'expert_bias_constraint': constraints.serialize(self.expert_bias_constraint),
            'gating_bias_constraint': constraints.serialize(self.gating_bias_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer)
        }
        base_config = super(DenseMoE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))