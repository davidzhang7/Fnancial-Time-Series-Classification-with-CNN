"""
Created on Wed Dec 19 22:21:17 2018

@author: Dawei Zhang (UNI: dz2363) 
"""


""" All functions needed to build CNN throughout researches
1. Dense_layer: wrapped building function of fully connected layer
2. Condensed_AlexNet: Parametrically reduced AlexNet
3. inception_mod: Building blocks of GoogleNet 
3. Condensed_GoogleNet: Parametrically reduced GoogleNet
4. VAE_alexnet: Variational autoencoder with AlexNet architecture on both ends
5. train_predict: function to train and make predictions from specified model


Remark: All hyperparameters are defined in file 'hyperparams.py'. Please do make it visible to this file!!!

"""

import numpy as np
import tensorflow as tf
import pandas as pd
import random
import time
from sklearn.metrics import confusion_matrix

from hyperparams import *
from data_processing_func import *

image_size = 16
channel_size = 4
numOfClass = 3

# input and output vector placeholders
x = tf.placeholder(tf.float32, [None, image_size*image_size*channel_size])
y = tf.placeholder(tf.float32, [None, numOfClass])

# A wrapped up dense layer function (defines dense layer operation)
def Dense_layer(inputs, 
                units, 
                name=None
                ):
    
    inputs_shape = tf.Tensor.get_shape(inputs).as_list()[-1]
    weight_matrix = tf.Variable(tf.truncated_normal([inputs_shape, units], stddev=0.01), name=name['weight'])
    bias = tf.Variable(tf.constant(1.0, shape=[units]), name=name['bias'])
    outputs = tf.matmul(inputs, weight_matrix)
    outputs = tf.nn.bias_add(outputs, bias)
    return outputs

# A lambda function to return proper size of bias added to conv layer
add_bias = lambda inputs: tf.nn.bias_add(inputs, tf.Variable(tf.constant(0.0, shape=[tf.Tensor.get_shape(inputs).as_list()[-1]])))

# Well-tuned condensed (parametrically reduced version) AlexNet
def Condensed_AlexNet(inputs, 
                      hyperparams=AlexNet_hyperparams, 
                      dropout=0.15
                      ):    
    """ A parametrically reduced version of AlexNet (For full version of AlexNet, see Krizhevsky (2012))
    
    Preserved the structure/architecture of AlexNet, but reduced in layers/hyperparameters

    """
    batch_size, _, image_size, channel_size = inputs.shape
    inputs = tf.reshape(inputs, [-1, image_size, image_size, channel_size])
    
    
    # 1st convolutional layer block
    x = tf.nn.conv2d(inputs, filter=hyperparams["conv1_conv_kernel"], strides=[1, 1, 1, 1], padding="SAME", name="conv1_conv")
    x = add_bias(x)
    x = tf.nn.tanh(x)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="conv1_maxP")

    
    # 2nd convolutional layer block
    x = tf.nn.conv2d(x, filter=hyperparams["conv2_conv_kernel"], strides=[1, 1, 1, 1], padding="SAME", name="conv2_conv")
    x = add_bias(x)
    x = tf.nn.tanh(x)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME",name="conv3_maxP")
    
    # 3rd convolutional layer block
    x = tf.nn.conv2d(x, filter=hyperparams["conv3_conv_kernel"], strides=[1, 1, 1, 1], padding="SAME", name="conv3_conv")
    x = add_bias(x)
    x = tf.nn.tanh(x)
    
    # 4th convolutional layer block
    x = tf.nn.conv2d(x, filter=hyperparams["conv4_conv_kernel"], strides=[1, 1, 1, 1], padding="SAME", name="conv4_conv")
    x = add_bias(x)
    x = tf.nn.tanh(x)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="VALID",name="conv4_maxP")
    
    
    # Flatten the inputs
    batch_size, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h*w*c])
    x = tf.nn.dropout(x, keep_prob=1-dropout)
    
    
    # 1st dense layer
    x = Dense_layer(x, units=150, name={"weight":"dense1_weight", "bias":"dense1_bias"})
    x = tf.nn.tanh(x)

    # 2nd dense layer
    x = Dense_layer(x, units=25, name={"weight":"dense2_weight", "bias":"dense2_bias"})
    x = tf.nn.tanh(x)
    
    x = Dense_layer(x, units=numOfClass, name={"weight":"dense3_weight", "bias":"dense3_bias"})
    outputs = tf.nn.softmax(x)

    # Return the outputs
    return x, outputs



# Well-tuned inception module used in the subsequent GoogLeNet-like CNN
def inception_mod(inputs, hyperparams, name=None):

    # 1x1 pathway  
    x1 = tf.nn.conv2d(inputs, filter=hyperparams["1x1_conv_kernel"], strides=[1, 1, 1, 1], padding='SAME', name="1x1_conv")
    x1 = add_bias(x1)
    x1 = tf.nn.tanh(x1)
    
    # 1x1 to 3x3 pathway
    x2 = tf.nn.conv2d(inputs, filter=hyperparams["3x3_conv_kernel1"], strides=[1, 1, 1, 1], padding='SAME', name="3x3_conv1")
    x2 = tf.nn.conv2d(x2, filter=hyperparams["3x3_conv_kernel2"], strides=[1, 1, 1, 1], padding='SAME', name="3x3_conv2")
    x2 = add_bias(x2)
    x2 = tf.nn.tanh(x2)
    
    # 1x1 to 5x5 pathway
    x3 = tf.nn.conv2d(inputs, filter=hyperparams["5x5_conv_kernel1"], strides=[1, 1, 1, 1], padding='SAME', name="5x5_conv1")
    x3 = tf.nn.conv2d(x3, filter=hyperparams["5x5_conv_kernel2"], strides=[1, 1, 1, 1], padding='SAME', name="5x5_conv2")
    x3 = add_bias(x3)
    x3 = tf.nn.tanh(x3)
    
    # 3x3 to 1x1 pathway
    x4 = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name="pooling1")
    x4 = tf.nn.conv2d(x4, filter=hyperparams['pooling1_conv_kernel'], strides=[1, 1, 1, 1], padding='SAME', name="pooling1_conv")
    x4 = add_bias(x4)
    x4 = tf.nn.tanh(x4)
    
    x = tf.concat([x1, x2, x3, x4], axis=3)  # Concat in the 4th dim to stack
    outputs = tf.tanh(x)
    
    return outputs

# Condensed GoogleNet
def Condensed_GoogleNet(inputs, 
                        hyperparams=GoogleNet_hyperparams, 
                        inception1_hyperparam=inception1_hyperparam, 
                        inception2_hyperparam=inception2_hyperparam, 
                        inception3_hyperparam=inception3_hyperparam,
                        dropout=0.0
                       ):
    
    """ A parametrically reduced version of GoogleNet-like CNN (For description and implementation of full GoogLeNet, see Szegedy et. al. (2015)) 
    
    This model preserves three layers of inception module. The realization is self-evident in the following codes.
    Hyperparameter lists for vanilla convolutional blocks and inception modules are defined in hyperparam_list.py
    
    """
    
    
    batch_size, _, image_size, channel_size = inputs.shape
    inputs = tf.reshape(inputs, [-1, image_size, image_size, channel_size])
    
    # 1st convolutional layer block
    x = tf.nn.conv2d(inputs, filter=hyperparams['conv1_conv_kernel'], strides=[1, 1, 1, 1], padding='SAME', name='conv1_conv')
    x = add_bias(x)
    x = tf.nn.tanh(x)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name='conv1_maxP')
    
    
    # 2nd convolutional layer block
    x = tf.nn.conv2d(x, filter=hyperparams['conv2_conv_kernel'], strides=[1, 1, 1, 1], padding="SAME", name='conv2_conv')
    x = add_bias(x) 
    x = tf.nn.tanh(x)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name='conv2_maxP')
    
    
    # 1st inception module
    x = inception_mod(x, inception1_hyperparam, name='inception1')
    x = tf.nn.max_pool(x, ksize=[1, 2, 2 ,1], strides=[1, 1, 1, 1], padding="SAME", name='inception1_maxP')
    
    
    # 2nd inception module
    x = inception_mod(x, inception2_hyperparam, name='inception2')
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name='inception2_maxP')
    
    
    # 3rd inception module
    x = inception_mod(x, inception3_hyperparam, name='inception3')
    x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="VALID", name='inception3_avgP')
    
    # Flatten and dropout
    x = tf.layers.flatten(x)
    x = tf.nn.dropout(x, keep_prob=1-dropout)
    
    x = Dense_layer(x, units=numOfClass, name={"weight":"dense1_weight", "bias":"dense1_bias"})
    outputs = tf.nn.softmax(x)
    
    return x, outputs



def VAE_alexnet(inputs, hyperparams=VAE_hyperparam, dropout=0.2, hidden_dim=150):
    
    batch_size, _, image_size, channel_size = inputs.shape
    inputs = tf.reshape(inputs, [-1, image_size, image_size, channel_size])
    
    # Encoding part
    x = tf.nn.conv2d(inputs, filter=hyperparams['encoder_conv1_kernel'], strides=[1, 1, 1 ,1], padding="SAME", name="encoder_conv1")
    x = add_bias(x)
    x = tf.nn.conv2d(x, filter=hyperparams['encoder_conv2_kernel'], strides=[1, 1, 1, 1], paddnig="SAME", name="encoder_conv2")
    x = add_bias(x)

    m,v = tf.nn.moments(x)
    x = tf.nn.batch_normalization(x, mean=m, variance=v, scale=None, offset=None, variance_epsilon=1e-7)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="encoder_maxpool")
    x = tf.nn.dropout(x, keep_prob=1-dropout)
    x = tf.layers.flatten(x)
    
    hidden, mu, log_sigma2 = Dense_layer(x, units=hidden_dim), Dense_layer(x, units=hidden_dim), Dense_layer(x, units=hidden_dim)
    
    # Decoding part
    x = Dense_layer(hidden, units=image_size*image_size*channel_size)
    x = tf.nn.dropout(x, keep_prob=1-dropout)
    x = tf.reshape(x, [-1, image_size, image_size, channel_size])
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="decoder_maxpool")
    x = tf.nn.conv2d(inputs, filter=hyperparams['decoder_conv1_kernel'], strides=[1, 1, 1 ,1], padding="SAME", name="encoder_conv1")
    x = add_bias(x)
    x = tf.nn.conv2d(x, filter=hyperparams['decoder_conv2_kernel'], strides=[1, 1, 1, 1], paddnig="SAME", name="encoder_conv2")
    output = add_bias(x)

    
    return output, hidden, mu, log_sigma2



# A training function for CNN, take in arguments specifying dataset as well as what CNN to use
def train_predict(X, Y, valX, valY, 
                  model, model_ckpt_path, cost_sensitive_loss=False,
                  cost_matrix=np.array([[0,1.1,1.3],[1.1,0,1.1],[1.3,1.1,0]]),
                  isBayesian=False,
                  batch_size=128, epochs=50,  
                  *args, **kwargs):
    
    """ Function to train and make prediction with models
    
        Args:
            X, Y, valX, valY: training set (X, Y), validation set (valX, valY). The function will return a prediction based
                              on the trained model on validation set valX
            model: specified model created with TensorFlow. In this research, will take either Condensed_AlexNet or
                   Condensed_Googlenet or Alex_VAE
            model_ckpt_path: path to save the model checkpoint
            cost_sensitive_loss: boolean, will implement cost sensitive cross entropy loss function if true. See Kukar and Kononenko (1998) for details.
            cost_matrix: a matrix of size output_dimension by output_dimension, default to be identity. (i,j)-th element is a 
                         penalty factor assigned for misclassifying i to j. It used to combine sample prior 
                         (distribution of classes in the training sample) to compute expected penalty of misclassifying i-th class.
                         See Kukar and Kononenko (1998) for details
            isBayesian: boolean, if true, will ensemble the result from the last 10 epochs of training
            batch_size, epochs, display_step: some paramters for training and displaying the model
            *args, **kwargs: arguments for functional input model. See arguments for the 3 types of models.
        
        Return:
            1. prediction of validation data based on current model
            2. Training and validation loss of current model
        
        Will save the session to prespecified model checkpoint path for reuse.
    """
    
    # Get sample distribution to calculate expected penalty
    sample_size, _, image_size, channel_size = X.shape
    val_size = valX.shape[0]
    _, numOfClass = Y.shape
    
    sample_prior = Y.sum(axis=0)/len(Y)
    expected_penalty = np.dot(cost_matrix, sample_prior) / (1-sample_prior)
    
    # Cost sensitive loss function with cross-entropy
    def cost_sensitive_loss(y_true, y_pred):
        y_pred = tf.clip_by_value((y_pred*expected_penalty)/tf.reduce_sum(y_pred*expected_penalty, axis=-1, keepdims=True), 1e-7, 1-1e-7)
        loss = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(loss)
    
#    def vae_loss(X, X_output, mu, log_sigma2):
#        reconstruction_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=X, predictions=X_output))
#        distribution_loss = -0.5*tf.reduce_mean(-tf.exp(log_sigma2) - tf.square(mu) + log_sigma2, axis=-1)
#        return tf.reduce_mean(reconstruction_loss+distribution_loss)

    # number of batches
    nB = int(sample_size/batch_size)+1
    val_nB = int(val_size/batch_size)+1
    
    x_batch = tf.placeholder(tf.float32, [None, image_size, image_size, channel_size])
    y_batch = tf.placeholder(tf.float32, [None, numOfClass])
    
    pre_thresholded_output, output = model(inputs=x_batch, **kwargs)
    
    # Create losses and other metrics
    if cost_sensitive_loss:
        xentropy_loss = cost_sensitive_loss(y_batch, output)
    else:
        xentropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_batch, logits=pre_thresholded_output))
        
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y_batch, axis=1)), tf.float32))
    
    # Create optimizer
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer().minimize(xentropy_loss, global_step=global_step)
    init = tf.global_variables_initializer()
    
    pred = []
    
    with tf.Session() as sess:
        sess.run(init)
        
        training_loss = []
        val_loss = []
        
        for e in range(epochs):
            
            epoch_start_time = time.time()
            
            # Shuffle dataset before doing an epoch of training
            Z = list(zip(X, Y))
            random.shuffle(Z)
            X, Y = zip(*Z)
            
            for b in range(nB):
                
                if (b+1)*batch_size>len(X):
                    x_b, y_b = X[b*batch_size:], Y[b*batch_size:]
                else:
                    x_b, y_b = X[b*batch_size:(b+1)*batch_size], Y[b*batch_size:(b+1)*batch_size]
                    
                _, step = sess.run([optimizer, global_step], feed_dict={x_batch:x_b, y_batch:y_b})
                
                        
            # Make evaluation on a batch basis
            cc, aa = 0, 0
            for b in range(nB):
                
                if (b+1)*batch_size>len(X):
                    x_b, y_b = X[b*batch_size:], Y[b*batch_size:]
                else:
                    x_b, y_b = X[b*batch_size:(b+1)*batch_size], Y[b*batch_size:(b+1)*batch_size]
                
                c, a = sess.run([xentropy_loss, accuracy], feed_dict={x_batch: x_b, y_batch: y_b})
                cc+=(c*len(y_b))
                aa+=(a*len(y_b))
                
            epoch_loss, epoch_acc = cc/(1.0*sample_size), aa/(1.0*sample_size)
            training_loss.append(epoch_loss)
            
            # Make validation on a batch basis
            vcc, vaa = 0, 0
            
            thisOutput = np.array([[0, 0, 0]])
            for b in range(val_nB):
                
                if (b+1)*batch_size>len(valX):
                    x_b, y_b = valX[b*batch_size:], valY[b*batch_size:]
                else:
                    x_b, y_b = valX[b*batch_size:(b+1)*batch_size], valY[b*batch_size:(b+1)*batch_size]
                
                o, c, a = sess.run([output, xentropy_loss, accuracy], feed_dict={x_batch: x_b, y_batch: y_b})
                thisOutput = np.append(thisOutput, o, axis=0)
                vcc+=(c*len(y_b))
                vaa+=(a*len(y_b))
            
            pred.append(thisOutput[1:])
            epoch_val_loss, epoch_val_acc = vcc/(1.0*val_size), vaa/(1.0*val_size)
            val_loss.append(epoch_val_loss)
            
            
            print("Training time: ", round(time.time()-epoch_start_time, 2),
                  'Epoch: {:03d}/{:03d} ======== Loss: {:.4f} Accuracy: {:.4f} val_Loss: {:.4f} val_acc: {:.4f} '.format(e+1, epochs, epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc))
        
        
        writer = tf.summary.FileWriter('logs', sess.graph)
        writer.close()
        
        saver = tf.train.Saver()
        saver.save(sess, model_ckpt_path)
        
        if isBayesian:
            # Sample from the last 10 epochs to form Bayesian learning: average across last 10 epochs of predictions
            return (np.mean(pred[-10:], axis=0), training_loss, val_loss)
        else:
            return (pred[-1], training_loss, val_loss)

        
def get_confusion_matrix(ytrue, ypred):
    
    # Create confusion matrix based on true results ytrue and predicted result ypred
    
    ytrue = np.argmax(ytrue, axis=1)
    ypred = np.argmax(ypred, axis=1)
    return confusion_matrix(ytrue, ypred)

