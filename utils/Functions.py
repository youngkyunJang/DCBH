import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.contrib.framework import arg_scope
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten, xavier_initializer

weight_decay = 0.0002
regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
initializer = xavier_initializer()

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        conv = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME', kernel_regularizer=regularizer, kernel_initializer=initializer)
        return  conv
def Linear(x, out_length, layer_name) :
    with tf.name_scope(layer_name):
        linear = tf.layers.dense(inputs=x, units=out_length, kernel_regularizer=regularizer, kernel_initializer=initializer)
        return linear
def Batch_Normalization(x, training, scope="batch"):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))
def Relu(x):
    return tf.nn.relu(x)
def LRelu(x, alpha=0.2):
    return tf.nn.relu(x) - alpha*tf.nn.relu(-x)
def Tanh(x):
    return tf.nn.tanh(x)
def SoftMax(x):
    return tf.nn.softmax(x)
def Flatten(x):
    return flatten(x)
def Dropout(x, prob, flag):
    if(flag==True):
        output = tf.nn.dropout(x, keep_prob=prob)
    else:
        output = tf.nn.dropout(x, keep_prob=1.0)
    return output
def Max_Pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)
def Global_Average_Pooling(x):
     return global_avg_pool(x, name='Global_avg_pooling')

def Slice_Encode(x, numBits, bias=0.5):
    concat_split = tf.split(x, numBits, 1)
    for i in range (numBits):
        if i == 0:
            concat_out = tf.reduce_max(tf.nn.softmax(concat_split[0])-bias, axis=1, keep_dims=True)
        else:
            concat_out = tf.concat([concat_out, (tf.reduce_max(tf.nn.softmax(concat_split[i])-bias, axis=1, keep_dims=True))], axis=1)
    return concat_out

def Slice_Encode_Linear(x, numBits):
    concat_split = tf.split(x, numBits, 1)
    for i in range (numBits):
        if i == 0:
            concat_out = tf.layers.dense(concat_split[0], 1, kernel_regularizer=regularizer, kernel_initializer=initializer)
        else:
            concat_out = tf.concat([concat_out, tf.layers.dense(concat_split[i], 1, kernel_regularizer=regularizer, kernel_initializer=initializer)], axis=1)
    return concat_out

def Q_loss(x, alpha=1.0):
    length = x.get_shape()[1]
    one_vector = tf.ones(length, dtype=tf.float32)*alpha
    l1dist = tf.abs((tf.subtract(one_vector, tf.abs(x))))
    tf.add_to_collection('quantization_loss', l1dist)

def C_loss(features, labels, num_classes):
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32, initializer=initializer, trainable=True)
    labels = tf.argmax(labels, axis=1)
    labels = tf.reshape(labels, [-1])

    centers_batch = tf.gather(centers, labels)

    loss = tf.reduce_mean(tf.nn.l2_loss(features - centers_batch))

    diff = centers_batch - features

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers_update_op
def Code_Matching(centers, features, numSeg, TwoClu, SplitLen, alpha):

    if TwoClu == 2:
        CluStr = '{0:02b}'
        numCluster = pow(2, TwoClu)
        for i in range(numCluster):
            tmp_bin = np.expand_dims(np.asarray(list(CluStr.format(i)), np.int32), 0)
            if i == 0:
                bin_list = tmp_bin
            else:
                bin_list = np.append(bin_list, tmp_bin, axis=0)

    elif TwoClu == 4:
        CluStr = '{0:04b}'
        numCluster = pow(2, TwoClu)
        for i in range(numCluster):
            tmp_bin = np.expand_dims(np.asarray(list(CluStr.format(i)), np.int32), 0)
            if i == 0:
                bin_list = tmp_bin
            else:
                bin_list = np.append(bin_list, tmp_bin, axis=0)
            #h_dist = np.sum(np.bitwise_xor(bin_min, tmp))

    elif TwoClu == 6:
        CluStr = '{0:06b}'
        numCluster = pow(2, TwoClu)
        for i in range(numCluster):
            tmp_bin = np.expand_dims(np.asarray(list(CluStr.format(i)), np.int32), 0)
            if i == 0:
                bin_list = tmp_bin
            else:
                bin_list = np.append(bin_list, tmp_bin, axis=0)

    else:
        CluStr = '{0:08b}'
        numCluster = pow(2, TwoClu)
        for i in range(numCluster):
            tmp_bin = np.expand_dims(np.asarray(list(CluStr.format(i)), np.int32), 0)
            if i == 0:
                bin_list = tmp_bin
            else:
                bin_list = np.append(bin_list, tmp_bin, axis=0)

    bin_list = tf.convert_to_tensor(bin_list, tf.float32)
    x = tf.split(features, numSeg, 1)
    y = tf.split(centers, numSeg, 1)
    for i in range(numSeg):
        size_x = tf.shape(x[i])[0]
        size_y = tf.shape(y[i])[0]
        xx = tf.expand_dims(x[i], -1)
        xx = tf.tile(xx, tf.stack([1, 1, size_y]))

        yy = tf.expand_dims(y[i], -1)
        yy = tf.tile(yy, tf.stack([1, 1, size_x]))
        yy = tf.transpose(yy, perm=[2, 1, 0])

        diff = tf.subtract(xx, yy)
        square_diff = tf.square(diff)
        square_diff = tf.reduce_sum(square_diff, 1)
        softmax_diff = tf.nn.softmax(square_diff*(-1*alpha))
        if i==0:
            arg = tf.argmax(softmax_diff, 1)
            idx = tf.reshape(arg, [-1, 1])
            y_split = tf.split(y[i], SplitLen, 1)
            for j in range(SplitLen):
                if j == 0:
                    soft_des_tmp = tf.transpose(tf.matmul(y_split[j],softmax_diff, transpose_a=True, transpose_b=True))
                else:
                    soft_des_tmp = tf.concat([soft_des_tmp, tf.transpose(tf.matmul(y_split[j],softmax_diff, transpose_a=True, transpose_b=True))], 1)
            descriptor = soft_des_tmp
            cluster_loss = tf.reshape(tf.reduce_min(square_diff, 1), [-1 ,1])
        else:
            arg = tf.argmax(softmax_diff, 1)
            idx = tf.concat([idx, tf.reshape(arg, [-1, 1])], axis=1)
            y_split = tf.split(y[i], SplitLen, 1)
            for j in range(SplitLen):
                if j == 0:
                    soft_des_tmp = tf.transpose(tf.matmul(y_split[j],softmax_diff, transpose_a=True, transpose_b=True))
                else:
                    soft_des_tmp = tf.concat([soft_des_tmp, tf.transpose(tf.matmul(y_split[j],softmax_diff, transpose_a=True, transpose_b=True))], 1)
            descriptor = tf.concat([descriptor, soft_des_tmp], axis=1)
            cluster_loss = tf.concat([cluster_loss, tf.reshape(tf.reduce_min(square_diff, 1), [-1 ,1])], axis=1)
    split_idx = tf.split(idx, numSeg, 1)
    for k in range(numSeg):
        if k == 0:
            bin_table = tf.squeeze(tf.gather(bin_list, split_idx[k]), axis=1)
        else:
            bin_table = tf.concat([bin_table, tf.squeeze(tf.gather(bin_list, split_idx[k]),axis=1)], axis=1)

    cluster_loss = tf.reduce_mean(cluster_loss, 1)
    return descriptor, cluster_loss, bin_table
def Shuffle_Table(Table, numSeg):
    Table_split = tf.split(Table, numSeg, 1)
    for i in range(numSeg):
        if i == 0:
            Table_shuffle = tf.random_shuffle(Table_split[i])
        else:
            Table_shuffle = tf.concat([Table_shuffle, tf.random_shuffle(Table_split[i])], 1)
    return Table_shuffle

def Evaluate(sess, x, label, training_flag, test_x, test_y, cost, accuracy, test_iteration=5):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0

    test_data_num = np.shape(test_x)[0]
    add = int(test_data_num/test_iteration)

    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)
        test_acc += acc_ / test_iteration
        test_loss += loss_ / test_iteration

    return test_acc, test_loss