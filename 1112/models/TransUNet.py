import tensorflow as tf

"""
Input, 
Layer, MultiHeadAttention, LayerNormalization, Dense, Embedding,
Maxpooling3D, AveragePooling3D, UpSampling3D, Conv3DTranspose, GlobalAveragePooling3D,
Conv3D,
BatchNormalization, Activation, concatenate, multiply, add,
ReLU, LeakyReLU, PReLU, ELU, Softmax
"""
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.layers import *
from utils.activation import *

def freeze_model(model, freeze_batch_norm=False):
    if freeze_batch_norm:
        for layer in model.layers:
            layer.trainable = False
            
    else:
        for layer in model.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
                
    return model

# ========================================================================================================================= #

def encoder_layer_3D(X,
                     channel,
                     pool_size,
                     pool,
                     kernel_size='auto',
                     activation = 'ReLU',
                     batch_norm = False,
                     name='encode'):
    if pool is True:
        pool = 'max'
        X = keras.layers.MaxPooling3D(pool_size=(pool_size, pool_size, pool_size),
                         name=f"{name}_maxpool")(X)
        
    elif pool is False:
        bias_flag = not batch_norm
        
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
            
        X = keras.layers.Conv3D(channel,
                   kernel_size,
                   strides=(pool_size, pool_size, pool_size),
                   padding='valid',
                   use_bias=bias_flag,
                   name=f"{name}_stride_conv")(X)
        
        if batch_norm:
            X = keras.layers.BatchNormalization(axis=4, name=f"{name}_bn")(X)
            
        if activation is not None:
            activation_func = eval(activation)
            X = activation_func(name=f"{name}_activation")(X)
            
    return X

def CONV_stack_3D(X,
                  channel,
                  kernel_size=3,
                  stack_num=2,
                  dilation_rate=1,
                  activation='ReLU',
                  batch_norm = False,
                  name='conv_block'):
    bias_flag = not batch_norm

    for i in range(stack_num):
        activation_func = eval(activation)
        
        X = keras.layers.Conv3D(channel,
                   kernel_size=kernel_size,
                   padding='same',
                   use_bias=bias_flag,
                   dilation_rate=dilation_rate,
                   name=f"{name}_{i}")(X)
        
        if batch_norm:
            X = keras.layers.BatchNormalization(axis=4, name=f"{name}_{i}_bn")(X)
            
        activation_func = eval(activation)
        X = activation_func(name=f"{name}_{i}_activation")(X)
    
    return X
        
def UNET_left_3D(X,
                 channel,
                 kernel_size,
                 stack_num,
                 activation,
                 pool,
                 batch_norm,
                 name):
    pool_size = 2
    
    X = encoder_layer_3D(X,
                         channel,
                         pool_size,
                         pool,
                         activation=activation,
                         batch_norm = batch_norm,
                         name = f"{name}_encode")
    X = CONV_stack_3D(X,
                      channel,
                      kernel_size=kernel_size,
                      stack_num=stack_num,
                      activation=activation,
                      batch_norm=batch_norm,
                      name=f"{name}_conv")
    
    return X

# ========================================================================================================================= #

def decode_layer_3D(X,
                    channel,
                    pool_size,
                    unpool,
                    kernel_size=3,
                    activation='ReLU',
                    batch_norm = False,
                    name='decode'):
    if unpool is False:
        bias_flag = not batch_norm
        
    elif unpool == 'nearest':
        unpool = True
        interp = 'nearest'
        
    elif (unpool is True) or (unpool == 'bilinear'):
        unpool=True
        
    if unpool:
        X = keras.layers.UpSampling3D(size=(pool_size, pool_size, pool_size),
                         name=f"{name}_unpool")(X)
        
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
            
        X = keras.layers.Conv3DTranspose(channel,
                            kernel_size,
                            strides=(pool_size, pool_size, pool_size),
                            padding='same',
                            name=f"{name}_trans_conv")(X)
        
        if batch_norm:
            X = keras.layers.BatchNormalization(axis=4, name=f"{name}_bn")(X)
            
        if activation is not None:
            activation_func = eval(activation)
            X = activation_func(name=f"{name}_activation")(X)
            
    return X

def UNET_right_3D(X,
                  X_list,
                  channel,
                  kernel_size=3,
                  stack_num=2,
                  activation='ReLU',
                  unpool=True,
                  batch_norm=False,
                  concat=True,
                  name='right0'):
    pool_size = 2
    
    X = decode_layer_3D(X, channel, pool_size, unpool,
                        activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))

    # linear convolutional layers before concatenation
    X = CONV_stack_3D(X, channel, kernel_size, stack_num=1, activation=activation,
                      batch_norm=batch_norm, name='{}_conv_before_concat'.format(name))
    
    if concat:
        X = keras.layers.concatenate([X,] + X_list, axis=4, name = name + '_concat')
        
    X = CONV_stack_3D(X,
                      channel,
                      kernel_size=kernel_size,
                      stack_num=stack_num,
                      activation=activation,
                      batch_norm=batch_norm,
                      name = name + '_conv_after_concat')
    
    return X

# ========================================================================================================================= #

class patch_extract(Layer):
    def __init__(self,
                 patch_size,
                 patch_stride):
        super(patch_extract, self).__init__()
        self.patch_size = patch_size
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[1]
        self.patch_size_z = patch_size[2]
        self.patch_stride = patch_stride
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        
        patches = tf.extract_volume_patches(input=images,
                                            ksizes=(1, self.patch_size_x,
                                                    self.patch_size_y,
                                                    self.patch_size_z, 1),
                                            strides = (1, self.patch_stride,
                                                       self.patch_stride,
                                                       self.patch_stride,1
                                                       ),
                                            padding='VALID',)
        
        patch_dim = patches.shape[-1]
        patch_num_x = patches.shape[1]
        patch_num_y = patches.shape[2]
        patch_num_z = patches.shape[3]
        patches = tf.reshape(patches,
                             (batch_size, patch_num_x * patch_num_y * patch_num_z, patch_dim))
        
        return patches
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'patch_size' : self.patch_size, })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class patch_embedding(Layer):
    def __init__(self,
                 num_patch,
                 embed_dim):
        super(patch_embedding, self).__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.proj = keras.layers.Dense(embed_dim)
        self.pos_embed = keras.layers.Embedding(input_dim=num_patch, 
                                   output_dim = embed_dim)
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        embed = self.proj(patch) + self.pos_embed(pos)
        
        return embed
    
def ViT_MLP(X,
            filter_num,
            activation='GELU',
            name='MLP'):
    activation_func = eval(activation)
    
    for i, f in enumerate(filter_num):
        X = keras.layers.Dense(f, name=f"{name}_{i}")(X)
        X = activation_func(name=f"{name}_activation_{i}")(X)
        
    return X

def ViT_block(V,
              num_heads,
              key_dim,
              filter_num_MLP,
              activation='GELU',
              name='ViT'):
    V_atten = V
    V_atten = keras.layers.LayerNormalization(name=f"{name}_layer_norm_1")(V_atten)
    V_atten = keras.layers.MultiHeadAttention(num_heads=num_heads,
                                 key_dim=key_dim,
                                 name=f"{name}_atten")(V_atten, V_atten)
    
    V_add = keras.layers.add([V_atten, V], name=f"{name}_skip_1")
    
    V_MLP = V_add
    V_MLP = keras.layers.LayerNormalization(name=f"{name}_layer_norm_2")(V_MLP)
    V_MLP = ViT_MLP(V_MLP, filter_num_MLP, activation, name=f"{name}_mlp")
    
    V_out = keras.layers.add([V_MLP, V_add], name=f"{name}_skip_2")
    
    return V_out

def transunet_3d_base(input_tensor,
                      filter_num,
                      patch_size=3,
                      patch_stride=2,
                      stack_num_down=2,
                      stack_num_up=2,
                      embed_dim=768,
                      num_mlp=3072,
                      num_heads=12,
                      num_transformer=12,
                      activation='ReLU',
                      mlp_activation='GELU',
                      batch_norm=False,
                      pool=True,
                      unpool=True,
                      backbone=None,
                      weights='imagenet',
                      freeze_backbone=True,
                      freeze_batch_norm=True,
                      name='transunet'):
    X_skip = []
    depth_ = len(filter_num)
    
    patch_size_x = patch_size
    patch_size_y = patch_size
    patch_size_z = patch_size
    
    input_size_x = input_tensor.shape[1]
    input_size_y = input_tensor.shape[2]
    input_size_z = input_tensor.shape[3]
    
    encode_size_x = input_size_x // 2 ** (depth_ - 1)
    encode_size_y = input_size_y // 2 ** (depth_ - 1)
    encode_size_z = input_size_z // 2 ** (depth_ - 1)
    
    key_dim = embed_dim
    
    filter_num_MLP = [num_mlp, embed_dim]
    
    if backbone is None:
        X = input_tensor
        
        X = CONV_stack_3D(X,
                          filter_num[0],
                          stack_num=stack_num_down,
                          activation=activation,
                          batch_norm=batch_norm,
                          name=f"{name}_down0")
        
        X_skip.append(X)
        
        for i, f in enumerate(filter_num[1:]):
            X = UNET_left_3D(X,
                             f,
                             kernel_size=3,
                             stack_num=stack_num_down,
                             activation=activation,
                             pool=pool,
                             batch_norm=batch_norm,
                             name=f"{name}_down{i+1}")
            X_skip.append(X)
            
    X = X_skip[-1]
    X_skip = X_skip[:-1]
    
    X = keras.layers.Conv3D(filter_num[-1], 
               1,
               padding='valid',
               use_bias=False,
               name=f"{name}_conv_trans_before")(X)

    X = patch_extract((patch_size_x, patch_size_y, patch_size_z), patch_stride)(X)
    num_patches = X.shape[-2]

    X = patch_embedding(num_patches, embed_dim)(X)
    
    for i in range(num_transformer):
        X = ViT_block(X,
                      num_heads=num_heads,
                      key_dim=key_dim,
                      filter_num_MLP=filter_num_MLP,
                      activation=mlp_activation,
                      name=f"{name}_ViT_{i}")
    
    X = keras.layers.Dense(encode_size_x * encode_size_y * encode_size_z * embed_dim / num_patches)(X)
    X = tf.reshape(X, (-1, encode_size_x, encode_size_y, encode_size_z, embed_dim))
    X = keras.layers.Conv3D(filter_num[-1], 1, padding='valid', use_bias=False, name=f"{name}_conv_trans_after")(X)
    X_skip.append(X)
    
    X_skip = X_skip[::-1] # reverse 거꾸로
    X = X_skip[0]
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)
    
    filter_num_decode = filter_num[:-1][::-1]
    
    for i in range(depth_decode):
        X = UNET_right_3D(X,
                          [X_decode[i], ],
                          filter_num_decode[i],
                          stack_num=stack_num_up,
                          activation = activation,
                          unpool = unpool,
                          batch_norm = batch_norm,
                          name=f"{name}_up{i}")
        
    if depth_decode <  depth_ - 1:
        for i in range(depth_ - depth_decode - 1):
            i_real = i + depth_decode
            X = UNET_right_3D(X,
                              None,
                              filter_num_decode[i_real],
                              stack_num = stack_num_up,
                              activation = activation,
                              unpool = unpool,
                              batch_norm=batch_norm,
                              concat=False,
                              name=f"{name}_up{i_real}")
            
    return X

def CONV_output_3D(X,
                   out_channel,
                   kernel_size=1,
                   activation='Softmax',
                   name='conv_output'):
    X = keras.layers.Conv3D(out_channel, 
               kernel_size,
               padding='same',
               use_bias=True,
               name=name)(X)
    
    if activation:
        if activation == 'Sigmoid':
            X = keras.layers.Activation('sigmoid', name=f"{name}_activation")(X)
            
        else:
            activation_func = eval(activation)
            X = activation_func(name=f"{name}_activation")(X)
            
    return X

def TransUNet3D(input_size,
                filter_num,
                out_channel,
                patch_size=4,
                patch_stride=2,
                stack_num_down=2,
                stack_num_up=2,
                embed_dim=768,
                num_mlp=2048,
                num_heads=8,
                num_transformer=4,
                activation='ReLU',
                mlp_activation='GELU',
                output_activation='Sigmoid',
                batch_norm=True,
                pool=True,
                unpool='bilinear',
                backbone=None,
                weights='imagenet',
                freeze_backbone=True,
                freeze_batch_norm=True,
                name='transunet'):
    IN = keras.layers.Input(input_size)
    
    X = transunet_3d_base(IN,
                          filter_num = filter_num,
                          patch_size = patch_size,
                          patch_stride = patch_stride,
                          stack_num_down = stack_num_down,
                          stack_num_up=stack_num_up,
                          embed_dim=embed_dim,
                          num_mlp=num_mlp,
                          num_heads= num_heads,
                          num_transformer=num_transformer,
                          activation=activation,
                          mlp_activation=mlp_activation,
                          batch_norm=batch_norm,
                          pool=pool,
                          unpool = unpool,
                          backbone=backbone,
                          weights=weights,
                          freeze_backbone=freeze_backbone,
                          freeze_batch_norm=freeze_batch_norm,
                          name=name)
    
    OUT = CONV_output_3D(X,
                         out_channel,
                         kernel_size=1,
                         activation=output_activation,
                         name=f"{name}_output")
    
    model = Model(inputs=[IN, ], outputs=[OUT, ], name=f"{name}_model")
    
    return model