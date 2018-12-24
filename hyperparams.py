"""
Created on Wed Dec 19 22:21:17 2018

@author: Dawei Zhang (UNI: dz2363) 
"""


""" All the hyperparameters needed in the research are defined here
"""

import tensorflow as tf

# Weight parameters as devised in the original research paper
AlexNet_hyperparams = {
    # 1st Conv Layer block
    "conv1_conv_kernel": tf.Variable(tf.glorot_uniform_initializer()((2, 2, 4, 6)), name="conv1_conv_kernel"),
    
    # 2nd Conv Layer block
    "conv2_conv_kernel": tf.Variable(tf.glorot_uniform_initializer()((2, 2, 6, 10)), name="conv2_conv_kernel"),
    # 3rd Conv Layer block
    "conv3_conv_kernel": tf.Variable(tf.glorot_uniform_initializer()((2, 2, 10, 8)), name="conv3_conv_kernel"),
    
    # 3rd Conv Layer block
    "conv4_conv_kernel": tf.Variable(tf.glorot_uniform_initializer()((2, 2, 8, 4)), name="conv4_conv_kernel"),
}

inception1_hyperparam={
    # 1x1 pathway
    "1x1_conv_kernel": tf.Variable(tf.glorot_uniform_initializer()((1, 1, 8, 6)), name="inception1_1x1_conv_kernel"),
    
    # 1x1 to 3x3 pathway,
    "3x3_conv_kernel1": tf.Variable(tf.glorot_uniform_initializer()((1, 1, 8, 6)), name="inception1_3x3_conv_kernel1"),
    "3x3_conv_kernel2": tf.Variable(tf.glorot_uniform_initializer()((3, 3, 6, 8)), name="inception1_3x3_conv_kernel2"),
    
    # 1x1 to 5x5 pathway
    "5x5_conv_kernel1": tf.Variable(tf.glorot_uniform_initializer()((1, 1, 8, 6)), name="inception1_3x3_conv_kernel1"),
    "5x5_conv_kernel2": tf.Variable(tf.glorot_uniform_initializer()((5, 5, 6, 8)), name="inception1_5x5_conv_kernel1"),
    
    # 3x3 to 1x1 pathway
    "pooling1_conv_kernel": tf.Variable(tf.glorot_uniform_initializer()((3, 3, 8, 3)), name="inception1_pooling1_conv_kernel")
}

# hyperparameter list for 2nd inception module
inception2_hyperparam={
    # 1x1 pathway
    "1x1_conv_kernel": tf.Variable(tf.glorot_uniform_initializer()((1, 1, 25, 4)), name="inception2_1x1_conv_kernel"),
    
    # 1x1 to 3x3 pathway
    "3x3_conv_kernel1": tf.Variable(tf.glorot_uniform_initializer()((1, 1, 25, 4)), name="inception2_3x3_conv_kernel1"),
    "3x3_conv_kernel2": tf.Variable(tf.glorot_uniform_initializer()((3, 3, 4, 6)), name="inception2_3x3_conv_kernel2"),
    
    # 1x1 to 5x5 pathway
    "5x5_conv_kernel1": tf.Variable(tf.glorot_uniform_initializer()((1, 1, 25, 4)), name="inception2_5x5_conv_kernel1"),
    "5x5_conv_kernel2": tf.Variable(tf.glorot_uniform_initializer()((5, 5, 4, 6)), name="inception2_5x5_conv_kernel2"),
    
    # 3x3 to 1x1 pathway
    "pooling1_conv_kernel": tf.Variable(tf.glorot_uniform_initializer()((3, 3, 25, 3)), name="inception2_pooling1_conv_kernel")
}

# hyperparameter list for 3rd inception module
inception3_hyperparam={
    # 1x1 pathway
    "1x1_conv_kernel": tf.Variable(tf.glorot_uniform_initializer()((1, 1, 19, 3)), name="inception3_1x1_conv_kernel"),
    
    # 1x1 to 3x3 pathway
    "3x3_conv_kernel1": tf.Variable(tf.glorot_uniform_initializer()((1, 1, 19, 3)), name="inception3_3x3_conv_kernel1"),
    "3x3_conv_kernel2": tf.Variable(tf.glorot_uniform_initializer()((3, 3, 3, 4)), name="inception3_3x3_conv_kernel2"),
    
    # 1x1 to 5x5 pathway
    "5x5_conv_kernel1": tf.Variable(tf.glorot_uniform_initializer()((1, 1, 19, 3)), name="inception3_5x5_conv_kernel1"),
    "5x5_conv_kernel2": tf.Variable(tf.glorot_uniform_initializer()((5, 5, 3, 4)), name="inception3_5x5_conv_kernel2"),
    
    # 3x3 to 1x1 pathway
    "pooling1_conv_kernel": tf.Variable(tf.glorot_uniform_initializer()((3, 3, 19, 3)), name="inception3_pooling1_conv_kernel")
}

# hyperparameter list for other convolutional parts in GoogLeNet
GoogleNet_hyperparams = {
    # 1st convolutional layer block
    "conv1_conv_kernel": tf.Variable(tf.glorot_uniform_initializer()((2, 2, 4, 16)), name="conv1_conv_kernel"),
    
    # 2nd convolutional layer block
    "conv2_conv_kernel": tf.Variable(tf.glorot_uniform_initializer()((2, 2, 16, 8)), name="conv2_conv_kernel")
}

VAE_hyperparam = {
    
    # encoder 1st convolutional layer
    "encoder_conv1_kernel": tf.Variable(tf.glorot_uniform_initializer()((2, 2, 4, 6)), name="encoder_conv1_kernel"),
    
    # decoder 2nd convolutional layer
    "encoder_conv2_kernel": tf.Variable(tf.glorot_uniform_initializer()((2, 2, 6, 8)), name="encoder_conv2_kernel"),
    
    # decoder 1st convolutional layer
    "decoder_conv1_kernel": tf.Variable(tf.glorot_uniform_initializer()((2, 2, 4, 8)), name="decoder_conv1_kernel"),
    
    # decoder 2nd convolutional layer
    "decoder_conv2_kernel": tf.Variable(tf.glorot_uniform_initializer()((2, 2, 8, 4)), name="decoder_conv1_kernel")
        
}
