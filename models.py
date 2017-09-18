import tensorflow as tf
import numpy as np
import random
import math
import time
tf.set_random_seed(123456789)
np.random.seed(123456789)
random.seed(123456789)



# def BPR ():


def SVD(user_batch, item_batch, rate_batch, user_num, item_num, r, dim = 5):
    bias_global = tf.get_variable("bias_global", shape=[])
    w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
    w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
    bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
    bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
    w_user = tf.get_variable("embd_user", shape=[user_num, dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
    w_item = tf.get_variable("embd_item", shape=[item_num, dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.02))

    embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")

    embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")
    infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
    infer = tf.add(infer, bias_global)
    infer = tf.add(infer, bias_user)
    infer = tf.add(infer, bias_item, name="svd_inference")

    regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")
    cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
    penalty = tf.constant(r, dtype=tf.float32, shape=[], name="l2")
    cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))

    return bias_user, infer, cost


# def SVDpp():



def Autorec (itemCount, h, r): 
    data = tf.placeholder(tf.float32, [None, itemCount])
    mask = tf.placeholder(tf.float32, [None, itemCount])

    scale = math.sqrt(6 / (itemCount + h))

    W1 = tf.Variable(tf.random_uniform([itemCount, h], -scale, scale))
    b1 = tf.Variable(tf.random_uniform([h], -scale, scale))
    mid = tf.nn.sigmoid(tf.matmul(data, W1) + b1)

    W2 = tf.Variable(tf.random_uniform([h, itemCount], -scale, scale))
    b2 = tf.Variable(tf.random_uniform([itemCount], -scale, scale))
    y = tf.matmul(mid, W2) + b2

    regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2)

    cost = tf.reduce_mean(tf.reduce_sum(tf.square((y - data) * mask), 1, keep_dims=True)
                          + r * regularizer)          
    return data, mask, y, cost



def DualAPR_tanh (itemCount, h, r, ran, alpha, beta):
    data_e = tf.placeholder(tf.float32, [None, itemCount])
    mask_e = tf.placeholder(tf.float32, [None, itemCount])
    data_i = tf.placeholder(tf.float32, [None, itemCount])
    mask_i = tf.placeholder(tf.float32, [None, itemCount])

    #length_negative = tf.placeholder(tf.float32)
    #length_positive = tf.placeholder(tf.float32)
    length_samples = tf.placeholder(tf.float32)
    scale = math.sqrt(6 / (itemCount + h))

    # explicit rating network
    W1_e = tf.Variable(tf.random_uniform([itemCount, h], -scale, scale))
    b1_e = tf.Variable(tf.random_uniform([h], -scale, scale))
    mid_e = tf.nn.sigmoid(tf.matmul(data_e, W1_e) + b1_e)

    W2_e = tf.Variable(tf.random_uniform([h, itemCount], -scale, scale))
    b2_e = tf.Variable(tf.random_uniform([itemCount], -scale, scale))
    y_e = tf.matmul(mid_e, W2_e) + b2_e

    # implicit feedback network
    W1_i = tf.Variable(tf.random_uniform([itemCount, h], -scale, scale))
    b1_i = tf.Variable(tf.random_uniform([h], -scale, scale))
    mid_i = tf.nn.sigmoid(tf.matmul(data_i, W1_i) + b1_i)

    W2_i = tf.Variable(tf.random_uniform([h, itemCount], -scale, scale))
    b2_i = tf.Variable(tf.random_uniform([itemCount], -scale, scale))
    y_i = tf.matmul(mid_i, W2_i) + b2_i
    
    # tanh, (e^(20(x-0.5))-1) / (e^(20(x-0.5))+1)
    approx_unintMask = tf.tanh( alpha*(y_i - beta) )
    
    AE_loss_i = tf.reduce_mean(tf.reduce_sum(tf.square((y_i - data_i) * mask_i), 1, keep_dims=True))
    AE_loss_e = tf.reduce_mean(tf.reduce_sum(tf.square((y_e - data_e) * mask_e), 1, keep_dims=True))
    RANK_loss = -(1/length_samples) * tf.reduce_mean(tf.reduce_sum(y_e*approx_unintMask, 1, keep_dims=True))
    regularizer = tf.reduce_mean(tf.nn.l2_loss(W1_e) + tf.nn.l2_loss(W2_e) + tf.nn.l2_loss(b1_e) + tf.nn.l2_loss(b2_e)+tf.nn.l2_loss(W1_i) + tf.nn.l2_loss(W2_i) + tf.nn.l2_loss(b1_i) + tf.nn.l2_loss(b2_i))

    cost = tf.reduce_mean(AE_loss_i + AE_loss_e + ran*RANK_loss + r*regularizer)

    return data_e, mask_e, data_i, mask_i, length_samples, y_e, cost    

def APR (itemCount, h, r, ran): 
    data = tf.placeholder(tf.float32, [None, itemCount])
    mask = tf.placeholder(tf.float32, [None, itemCount])
  
    negativeMask = tf.placeholder(tf.float32, [None, itemCount])
    positiveMask = tf.placeholder(tf.float32, [None, itemCount])
    
    length_negative = tf.placeholder(tf.float32)
    length_positive = tf.placeholder(tf.float32)
    
    scale = math.sqrt(6 / (itemCount + h))

    W1 = tf.Variable(tf.random_uniform([itemCount, h], -scale, scale))
    b1 = tf.Variable(tf.random_uniform([h], -scale, scale))
    mid = tf.nn.sigmoid(tf.matmul(data, W1) + b1)

    W2 = tf.Variable(tf.random_uniform([h, itemCount], -scale, scale))
    b2 = tf.Variable(tf.random_uniform([itemCount], -scale, scale))
    
    y = tf.matmul(mid, W2) + b2

    AE_loss = tf.reduce_mean(tf.reduce_sum(tf.square((y - data) * mask), 1, keep_dims=True))
    RANK_loss = (1/length_positive)*tf.reduce_mean(tf.reduce_sum(y*negativeMask, 1, keep_dims=True)) - (1/length_negative)*tf.reduce_mean(tf.reduce_sum(y*positiveMask, 1, keep_dims=True))
    regularizer = tf.reduce_mean(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2))
    
    cost = tf.reduce_mean(AE_loss + ran*RANK_loss + r*regularizer)

    return data, mask, positiveMask, negativeMask, length_positive, length_negative, y, cost

    
def DualAPR_sigmoid (itemCount, h, r, ran, alpha, beta): 
    data_e = tf.placeholder(tf.float32, [None, itemCount])
    mask_e = tf.placeholder(tf.float32, [None, itemCount])
    data_i = tf.placeholder(tf.float32, [None, itemCount])
    mask_i = tf.placeholder(tf.float32, [None, itemCount])
    
    positiveMask = tf.placeholder(tf.float32, [None, itemCount])
    
    #length_negative = tf.placeholder(tf.float32)
    #length_positive = tf.placeholder(tf.float32)
    length_samples = tf.placeholder(tf.float32)
    scale = math.sqrt(6 / (itemCount + h))

    # explicit rating network
    W1_e = tf.Variable(tf.random_uniform([itemCount, h], -scale, scale))
    b1_e = tf.Variable(tf.random_uniform([h], -scale, scale))
    mid_e = tf.nn.sigmoid(tf.matmul(data_e, W1_e) + b1_e)

    W2_e = tf.Variable(tf.random_uniform([h, itemCount], -scale, scale))
    b2_e = tf.Variable(tf.random_uniform([itemCount], -scale, scale))
    y_e = tf.matmul(mid_e, W2_e) + b2_e

    # implicit feedback network
    W1_i = tf.Variable(tf.random_uniform([itemCount, h], -scale, scale))
    b1_i = tf.Variable(tf.random_uniform([h], -scale, scale))
    mid_i = tf.nn.sigmoid(tf.matmul(data_i, W1_i) + b1_i)

    W2_i = tf.Variable(tf.random_uniform([h, itemCount], -scale, scale))
    b2_i = tf.Variable(tf.random_uniform([itemCount], -scale, scale))
    y_i = tf.matmul(mid_i, W2_i) + b2_i
    
    # hard sigmoid, 1-(1/(1+1/e^(20(x-0.7))))
    approx_unintMask = 1 - tf.sigmoid( alpha*(y_i - beta) )
    AE_loss_i = tf.reduce_mean(tf.reduce_sum(tf.square((y_i - data_i) * mask_i), 1, keep_dims=True))
    AE_loss_e = tf.reduce_mean(tf.reduce_sum(tf.square((y_e - data_e) * mask_e), 1, keep_dims=True))

    RANK_loss = (-1/length_samples)*(tf.reduce_mean(tf.reduce_sum(y_e*positiveMask, 1, keep_dims=True)) - tf.reduce_mean(tf.reduce_sum(y_e*approx_unintMask, 1, keep_dims=True)))
    
    regularizer = tf.reduce_mean(tf.nn.l2_loss(W1_e) + tf.nn.l2_loss(W2_e) + tf.nn.l2_loss(b1_e) + tf.nn.l2_loss(b2_e)+tf.nn.l2_loss(W1_i) + tf.nn.l2_loss(W2_i) + tf.nn.l2_loss(b1_i) + tf.nn.l2_loss(b2_i))
    
    cost = tf.reduce_mean(AE_loss_i + AE_loss_e + ran*RANK_loss + r*regularizer)

    return data_e, mask_e, data_i, mask_i, positiveMask, length_samples, y_e, cost    
