import numpy as np
import tensorflow as tf
import os

def weight_variable(shape,name="weights"):
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
    return tf.Variable(initial,name= name)

def bias_variable(shape,name = "biases"):
    initial = tf.constant(0.1, dtype=tf.float32, shape= shape)
    return tf.Variable(initial, name = name)

def conv2d(input, w):
    return tf.nn.conv2d(input, w, [1,1,1,1],padding='SAME')

def max_pool(input):
    return tf.nn.max_pool(input,
                          ksize=[1,2,2,1],
                          strides=[1,2,2,1],
                          padding='SAME',
                          name='pool')

def batch_normal(input,name,activation_fn=None,is_training=True,scale=0):
    with tf.variable_scope(name) as scope:
        output = tf.contrib.layers.batch_norm(
            input,
            activation_fn=activation_fn,
            is_training=is_training,
            updates_collections=None,
            scale=scale,
            scope=scope)
        return output

def dropout(input, keep_prob, name):
    return tf.nn.dropout(input, keep_prob, name=name)

def liner(input, number, name):
    with tf.name_scope(name) as scope:
        kernel = weight_variable([2048,number])
        bias = bias_variable([number])
        out = tf.matmul(input, kernel) + bias
        return out


def conv_bn(input ,weight_shape,name ):
    with tf.name_scope(name) as scope:
        kernel = weight_variable(weight_shape)
        bias   = bias_variable([weight_shape[-1]])
        output_conv = conv2d(input,kernel)+bias
        # print(scope.name)
        bn = batch_normal(output_conv, name, activation_fn=tf.nn.relu, is_training=True)
        return bn

def residual(input,weight_shape, name):
    out1 = conv_bn(input, weight_shape, name[0])
    out2 = conv_bn(out1, weight_shape, name[1])
    return tf.add(input,out2)


def resnet9(x,num_classes):
    prep = conv_bn(x, [3,3,3,64], "prep")

    with tf.name_scope("layer1") as scope:
        layer1_head = conv_bn(prep, [3,3,64,128], "head1")
        pool1 = max_pool(layer1_head)
        layer1_res = residual(pool1,[3,3,128,128],["res1_1","res1_2"])

    with tf.name_scope("layer2") as scope:
        layer2 = conv_bn(layer1_res, [3,3,128,256],"head2")
        pool2  = max_pool(layer2)

    with tf.name_scope("layer3") as scope:
        layer3_head = conv_bn(pool2,[3,3,256,512],"head3")
        pool3 = max_pool(layer3_head)
        layer3_res = residual(pool3, [3,3,512,512],["res3_1","res3_2"])

    with tf.name_scope("classification") as scope:
        pool4 = max_pool(layer3_res)
        reshape_pool4 = tf.reshape(pool4,[-1,2048])
        Flatten = liner(reshape_pool4,num_classes,"fc")
    return Flatten




