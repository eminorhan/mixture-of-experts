import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from DenseMoE import DenseMoE
from scipy.io import savemat

# Function for computing the Jacobian
def jacobian(y, x):
    jacobian_flat = tf.stack( [tf.gradients(y_i, x)[0] for y_i in tf.unstack(y,axis=1)], axis=1)
    return jacobian_flat

n_data = 250
n_inp_dim = 128
n_hid_dim = 128

## Random normal input
x = np.random.rand(n_data,n_inp_dim)

which_model='moe' # 'dense' or 'moe'

input_shape = x.shape[1:]
inputs = Input(shape=input_shape)

if which_model=='moe':
    n_experts = 20
    hidden = DenseMoE(n_hid_dim, n_experts, expert_activation='relu', gating_activation='softmax')(inputs)
elif which_model=='dense':
    n_experts = 0 # dummy
    hidden = Dense(n_hid_dim, activation='relu')(inputs)

model = Model(inputs=inputs, outputs=hidden)

## Compute Jacobian
J = jacobian(model.output, model.input)
jacobian_func = K.function([model.input], [J])
j_vals = jacobian_func([x])[0]

## Compute Jacobian singular values
s_vals = np.linalg.svd(j_vals,compute_uv=False)
s_mean = np.mean(s_vals,axis=0)

savemat((which_model+'%i'+'_softmax.mat')%n_experts,{'s_mean':s_mean})

