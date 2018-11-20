import numpy as np
import cv2
import tensorflow as tf
import os
from preprocess import create_feed_dict, get_datasets, create_eval_feed_dict
import math
import random
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr

first_part_path = '../DIQA_Release_1.0_Part1'
second_part_path = '../DIQA_Release_1.0_Part2/FineReader/'
IMAGE_SIZE = 48
LR = 0.001
num_epoch = 10
batch_size = 6400

def conv_bn_relu(current, number, in_channels, out_channels, is_training, init):
    filters = tf.get_variable(name='conv' + str(number) + '_' + 'W',
                              initializer=init, shape=(5, 5, in_channels, out_channels))
    bias = tf.get_variable(name='conv' + str(number) + '_' + 'b',
                           initializer=init, shape=(out_channels))
    current = tf.nn.bias_add(tf.nn.conv2d(current, filters, strides=[1, 1, 1, 1], padding="VALID"), bias)

    batch_mean, batch_var = tf.nn.moments(current, [0, 1, 2], name='batch_moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
    offset = tf.Variable(tf.constant(0.0, shape=[out_channels]), name='offset' + str(number), trainable=True)
    scale = tf.Variable(tf.constant(1.0, shape=[out_channels]), name='scale' + str(number), trainable=True)
    current = tf.nn.batch_normalization(current, mean, variance, offset, scale, 1e-5)
    # current = tf.nn.relu(current)
    return current

def forward(image_patches, batch_size, sess, fc3, image_placeholder, is_training, keep_prob):
    print('image_patches:', image_patches.shape)
    nr_of_examples = len(image_patches)
    nr_of_batches = math.ceil(nr_of_examples / batch_size)
    patch_scores = np.zeros(nr_of_examples)
    batch_index = 0
    for batch_index in range(nr_of_batches - 1):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        fc3_ = sess.run(fc3, feed_dict={image_placeholder: image_patches[start_index:end_index], is_training: False, keep_prob: 1.})
        print('fc3_:', fc3_)
        patch_scores[start_index:end_index] = fc3_
    # batch_index += 1
    start_index = batch_index * batch_size
    fc3_ = sess.run(fc3, feed_dict={image_placeholder: image_patches[start_index:], is_training: False, keep_prob: 1.})
    patch_scores[start_index:] = fc3_
    print(image_patches[start_index:].shape)
    return np.mean(patch_scores)

def main():
    image_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 1), name='image_placeholder')
    label_placeholder = tf.placeholder(dtype=tf.float32, shape=(None), name='label_placeholder')
    is_training = tf.placeholder(tf.bool, name="is_training")
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    init = tf.contrib.layers.xavier_initializer()

    conv_1 = conv_bn_relu(image_placeholder, 1, 1, 40, is_training, init)

    max_pooled_1 = tf.nn.max_pool(conv_1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="VALID")

    conv_2 = conv_bn_relu(max_pooled_1, 2, 40, 80, is_training, init)

    max_pooled_2 = tf.squeeze(tf.nn.max_pool(conv_2, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding="VALID"), axis=(1, 2))
    min_pooled = tf.math.reduce_min(conv_2, axis=(1, 2))
    # concat_tensor = tf.concat([max_pooled_2, min_pooled], axis=1)

    fc_w1 = tf.Variable(tf.random_normal([80, 1024]))
    fc_b1 = tf.Variable(tf.random_normal([1024]))
    fc1 = tf.nn.relu(tf.add(tf.add(tf.matmul(max_pooled_2, fc_w1), fc_b1), tf.add(tf.matmul(min_pooled, fc_w1), fc_b1)))
    fc1 = tf.nn.dropout(fc1, keep_prob)

    fc_w2 = tf.Variable(tf.random_normal([1024, 1024]))
    fc_b2 = tf.Variable(tf.random_normal([1024]))
    fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, fc_w2), fc_b2))
    fc2 = tf.nn.dropout(fc2, keep_prob)

    fc_w3 = tf.Variable(tf.random_normal([1024, 1]))
    fc_b3 = tf.Variable(tf.random_normal([1]))
    fc3 = tf.squeeze((tf.add(tf.matmul(fc2, fc_w3), fc_b3)), axis=1)

    loss = tf.reduce_mean(tf.math.abs(tf.math.subtract(fc3, label_placeholder)))
    # loss = tf.losses.absolute_difference(label_placeholder, fc3)
    optimiser = tf.train.AdamOptimizer(learning_rate=LR)
    train_op = optimiser.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    training_image_paths, training_eval_paths, validation_image_paths, validation_eval_paths, test_image_paths, test_eval_paths = get_datasets(first_part_path, second_part_path)
    training_patches, training_scores = create_feed_dict(training_image_paths, training_eval_paths)
    
    nr_of_training_examples = len(training_scores)
    nr_of_training_batches = math.ceil(nr_of_training_examples / batch_size)

    eval_training_patches, eval_training_scores = create_eval_feed_dict(training_image_paths, training_eval_paths)
    
    
    validation_patches, validation_scores = create_eval_feed_dict(validation_image_paths, validation_eval_paths)
    test_patches, test_scores = create_eval_feed_dict(test_image_paths, test_eval_paths)

    # random.seed(3796)
    patch_indices = list(range(nr_of_training_examples))
    for epoch_index in range(num_epoch):
        random.shuffle(patch_indices)
        training_patches = training_patches[np.array(patch_indices)]
        training_scores = training_scores[np.array(patch_indices)]
        for batch_index in range(nr_of_training_batches - 1):
            

            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            loss_, fc3_, _ = sess.run([loss, fc3, train_op], feed_dict={image_placeholder: training_patches[start_index:end_index], label_placeholder: training_scores[start_index:end_index], is_training: True, keep_prob: 1.})
            print("Epoch:", epoch_index + 1, "Batch:", batch_index + 1, '/', nr_of_training_batches, 'Loss:', loss_)

            # print("predicted training:", fc3_)
            # print("gt training:", eval_training_scores)
        batch_index += 1
        start_index = batch_index * batch_size
        loss_, _ = sess.run([loss, train_op], feed_dict={image_placeholder: training_patches[start_index:], label_placeholder: training_scores[start_index:], is_training: True, keep_prob: 1.})
        print("Epoch:", epoch_index + 1, "Batch:", batch_index + 1, '/', nr_of_training_batches, 'Loss:', loss_)
        predicted_training_scores = np.zeros_like(eval_training_scores)
        for i in range(len(eval_training_patches)):
            predicted_training_scores[i] = forward(eval_training_patches[i], batch_size, sess, fc3, image_placeholder, is_training, keep_prob)
        print("Training LCC:", pearsonr(predicted_training_scores, eval_training_scores)[0])
        print("Training SROCC:", spearmanr(predicted_training_scores, eval_training_scores)[0])
        # print("predicted training:", predicted_training_scores)
        # print("gt training:", eval_training_scores)

        predicted_validation_scores = np.zeros_like(validation_scores)
        for i in range(len(validation_patches)):
            predicted_validation_scores[i] = forward(validation_patches[i], batch_size, sess, fc3, image_placeholder, is_training, keep_prob)
        print("Validation LCC:", pearsonr(predicted_validation_scores, validation_scores)[0])
        print("Validation SROCC:", spearmanr(predicted_validation_scores, validation_scores)[0])
        # print("predicted validation:", predicted_validation_scores)
        # print("gt validation:", validation_scores)


    predicted_test_scores = np.zeros_like(test_scores)
    for i in range(len(test_patches)):
        predicted_test_scores[i] = forward(test_patches[i], batch_size, sess, fc3, image_placeholder, is_training, keep_prob)
    print("Test LCC:", pearsonr(predicted_test_scores, test_scores)[0])
    print("Test SROCC:", spearmanr(predicted_test_scores, test_scores)[0])
    # print("predicted test:", predicted_test_scores)
    # print("gt test:", test_scores)            

if __name__ == '__main__':
    main()
