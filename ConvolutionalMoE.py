# -*- coding: utf-8 -*-
"""Convolutional MoE layers. The code here is based on the implementation of the standard convolutional layers in Keras.
"""
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.initializers import RandomNormal
from keras.engine.topology import Layer, InputSpec
from keras.utils import conv_utils

class _ConvMoE(Layer):
    """Abstract nD convolution layer mixture of experts (private, used as implementation base).
    """

    def __init__(self, rank,
                 n_filters,
                 n_experts_per_filter,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=1,
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
        super(_ConvMoE, self).__init__(**kwargs)
        self.rank = rank
        self.n_filters = n_filters
        self.n_experts_per_filter = n_experts_per_filter
        self.n_total_filters = self.n_filters * self.n_experts_per_filter
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')

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
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        expert_init_std = self.expert_kernel_initializer_scale / np.sqrt(input_dim*np.prod(self.kernel_size))
        gating_init_std = self.gating_kernel_initializer_scale / np.sqrt(np.prod(input_shape[1:]))

        expert_kernel_shape = self.kernel_size + (input_dim, self.n_total_filters)
        self.expert_kernel = self.add_weight(shape=expert_kernel_shape,
                                      initializer=RandomNormal(mean=0., stddev=expert_init_std),
                                      name='expert_kernel',
                                      regularizer=self.expert_kernel_regularizer,
                                      constraint=self.expert_kernel_constraint)

        gating_kernel_shape = input_shape[1:] + (self.n_filters, self.n_experts_per_filter)
        self.gating_kernel = self.add_weight(shape=gating_kernel_shape,
                                      initializer=RandomNormal(mean=0., stddev=gating_init_std),
                                      name='gating_kernel',
                                      regularizer=self.gating_kernel_regularizer,
                                      constraint=self.gating_kernel_constraint)

        if self.use_expert_bias:

            expert_bias_shape = ()
            for i in range(self.rank):
                expert_bias_shape = expert_bias_shape + (1,)
            expert_bias_shape = expert_bias_shape + (self.n_filters, self.n_experts_per_filter)

            self.expert_bias = self.add_weight(shape=expert_bias_shape,
                                        initializer=self.expert_bias_initializer,
                                        name='expert_bias',
                                        regularizer=self.expert_bias_regularizer,
                                        constraint=self.expert_bias_constraint)
        else:
            self.expert_bias = None

        if self.use_gating_bias:
            self.gating_bias = self.add_weight(shape=(self.n_filters, self.n_experts_per_filter),
                                        initializer=self.gating_bias_initializer,
                                        name='gating_bias',
                                        regularizer=self.gating_bias_regularizer,
                                        constraint=self.gating_bias_constraint)
        else:
            self.gating_bias = None

        self.o_shape = self.compute_output_shape(input_shape=input_shape)
        self.new_gating_outputs_shape = (-1,)
        for i in range(self.rank):
            self.new_gating_outputs_shape = self.new_gating_outputs_shape + (1,)
        self.new_gating_outputs_shape = self.new_gating_outputs_shape + (self.n_filters, self.n_experts_per_filter)

        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        if self.rank == 1:
            expert_outputs = K.conv1d(
                inputs,
                self.expert_kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            expert_outputs = K.conv2d(
                inputs,
                self.expert_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            expert_outputs = K.conv3d(
                inputs,
                self.expert_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        expert_outputs = K.reshape(expert_outputs, (-1,) + self.o_shape[1:-1] + (self.n_filters, self.n_experts_per_filter))

        if self.use_expert_bias:
            expert_outputs = K.bias_add(
                expert_outputs,
                self.expert_bias,
                data_format=self.data_format)

        if self.expert_activation is not None:
            expert_outputs = self.expert_activation(expert_outputs)

        gating_outputs = tf.tensordot(inputs, self.gating_kernel, axes=self.rank+1) # samples x n_filters x n_experts_per_filter

        if self.use_gating_bias:
            gating_outputs = K.bias_add(
                gating_outputs,
                self.gating_bias,
                data_format=self.data_format)

        if self.gating_activation is not None:
            gating_outputs = self.gating_activation(gating_outputs)

        gating_outputs = K.reshape(gating_outputs, self.new_gating_outputs_shape)
        outputs = K.sum(expert_outputs * gating_outputs, axis=-1, keepdims=False)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.n_filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.n_filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'n_filters': self.n_filters,
            'n_experts_per_filter':n_experts_per_filter,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'expert_activation': activations.serialize(self.expert_activation),
            'gating_activation': activations.serialize(self.gating_activation),
            'use_expert_bias': self.use_expert_bias,
            'use_gating_bias': self.use_gating_bias,
            'expert_kernel_initializer_scale':self.expert_kernel_initializer_scale,
            'gating_kernel_initializer_scale':self.gating_kernel_initializer_scale,
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
        base_config = super(_ConvMoE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Conv1DMoE(_ConvMoE):
    """1D convolution layer (e.g. temporal convolution).

    # Input shape
        3D tensor with shape: `(batch_size, steps, input_dim)`

    # Output shape
        3D tensor with shape: `(batch_size, new_steps, n_filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(self,
                 n_filters,
                 n_experts_per_filter,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=1,
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
        if padding == 'causal':
            if data_format != 'channels_last':
                raise ValueError('When using causal padding in `Conv1DMoE`, `data_format` must be "channels_last" (temporal data).')
        super(Conv1DMoE, self).__init__(
            rank=1,
            n_filters=n_filters,
            n_experts_per_filter=n_experts_per_filter,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            expert_activation=expert_activation,
            gating_activation=gating_activation,
            use_expert_bias=use_expert_bias,
            use_gating_bias=use_gating_bias,
            expert_kernel_initializer_scale=expert_kernel_initializer_scale,
            gating_kernel_initializer_scale=gating_kernel_initializer_scale,
            expert_bias_initializer=expert_bias_initializer,
            gating_bias_initializer=gating_bias_initializer,
            expert_kernel_regularizer=expert_kernel_regularizer,
            gating_kernel_regularizer=gating_kernel_regularizer,
            expert_bias_regularizer=expert_bias_regularizer,
            gating_bias_regularizer=gating_bias_regularizer,
            expert_kernel_constraint=expert_kernel_constraint,
            gating_kernel_constraint=gating_kernel_constraint,
            expert_bias_constraint=expert_bias_constraint,
            gating_bias_constraint=gating_bias_constraint,
            activity_regularizer=activity_regularizer,
            **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def get_config(self):
        config = super(Conv1DMoE, self).get_config()
        config.pop('rank')
        return config


class Conv2DMoE(_ConvMoE):
    """2D convolution layer (e.g. spatial convolution over images).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, rows, cols, channels)`
        if `data_format` is `"channels_last"`.

    # Output shape
        4D tensor with shape:
        `(samples, n_filters, new_rows, new_cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, n_filters)`
        if `data_format` is `"channels_last"`.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 n_filters,
                 n_experts_per_filter,
                 kernel_size,
                 strides=(1,1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1,1),
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
        super(Conv2DMoE, self).__init__(
            rank=2,
            n_filters=n_filters,
            n_experts_per_filter=n_experts_per_filter,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            expert_activation=expert_activation,
            gating_activation=gating_activation,
            use_expert_bias=use_expert_bias,
            use_gating_bias=use_gating_bias,
            expert_kernel_initializer_scale=expert_kernel_initializer_scale,
            gating_kernel_initializer_scale=gating_kernel_initializer_scale,
            expert_bias_initializer=expert_bias_initializer,
            gating_bias_initializer=gating_bias_initializer,
            expert_kernel_regularizer=expert_kernel_regularizer,
            gating_kernel_regularizer=gating_kernel_regularizer,
            expert_bias_regularizer=expert_bias_regularizer,
            gating_bias_regularizer=gating_bias_regularizer,
            expert_kernel_constraint=expert_kernel_constraint,
            gating_kernel_constraint=gating_kernel_constraint,
            expert_bias_constraint=expert_bias_constraint,
            gating_bias_constraint=gating_bias_constraint,
            activity_regularizer=activity_regularizer,
            **kwargs)
        self.input_spec = InputSpec(ndim=4)

    def get_config(self):
        config = super(Conv2DMoE, self).get_config()
        config.pop('rank')
        return config


class Conv3DMoE(_ConvMoE):
    """3D convolution layer (e.g. spatial convolution over volumes).

    # Input shape
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)`
        if `data_format` is `"channels_first"`
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)`
        if `data_format` is `"channels_last"`.

    # Output shape
        5D tensor with shape:
        `(samples, n_filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)`
        if `data_format` is `"channels_first"`
        or 5D tensor with shape:
        `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, n_filters)`
        if `data_format` is `"channels_last"`.
        `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have changed due to padding.
    """

    def __init__(self,
                 n_filters,
                 n_experts_per_filter,
                 kernel_size,
                 strides=(1,1,1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1,1,1),
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
        super(Conv3DMoE, self).__init__(
            rank=3,
            n_filters=n_filters,
            n_experts_per_filter=n_experts_per_filter,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            expert_activation=expert_activation,
            gating_activation=gating_activation,
            use_expert_bias=use_expert_bias,
            use_gating_bias=use_gating_bias,
            expert_kernel_initializer_scale=expert_kernel_initializer_scale,
            gating_kernel_initializer_scale=gating_kernel_initializer_scale,
            expert_bias_initializer=expert_bias_initializer,
            gating_bias_initializer=gating_bias_initializer,
            expert_kernel_regularizer=expert_kernel_regularizer,
            gating_kernel_regularizer=gating_kernel_regularizer,
            expert_bias_regularizer=expert_bias_regularizer,
            gating_bias_regularizer=gating_bias_regularizer,
            expert_kernel_constraint=expert_kernel_constraint,
            gating_kernel_constraint=gating_kernel_constraint,
            expert_bias_constraint=expert_bias_constraint,
            gating_bias_constraint=gating_bias_constraint,
            activity_regularizer=activity_regularizer,
            **kwargs)
        self.input_spec = InputSpec(ndim=5)

    def get_config(self):
        config = super(Conv3DMoE, self).get_config()
        config.pop('rank')
        return config


# Aliases
Convolution1DMoE = Conv1DMoE
Convolution2DMoE = Conv2DMoE
Convolution3DMoE = Conv3DMoE