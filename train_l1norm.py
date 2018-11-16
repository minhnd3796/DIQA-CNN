import numpy as np
import tensorflow as tf
import os
from preprocess import create_feed_dict, get_datasets
import math
import random

first_part_path = '../DIQA_Release_1.0_Part1'
second_part_path = '../DIQA_Release_1.0_Part2/FineReader/'
IMAGE_SIZE = 48
LR = 0.00001
num_epoch = 10
batch_size = 512

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return x

def main():
    image_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 1), name='image_placeholder')
    label_placeholder = tf.placeholder(dtype=tf.float32, shape=(None), name='label_placeholder')

    w1 = tf.Variable(tf.random_normal([5, 5, 1, 40]))
    b1 = tf.Variable(tf.random_normal([40]))
    conv_1 = conv2d(image_placeholder, w1, b1)

    max_pooled_1 = tf.nn.max_pool(conv_1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="VALID")

    w2 = tf.Variable(tf.random_normal([5, 5, 40, 80]))
    b2 = tf.Variable(tf.random_normal([80]))
    conv_2 = conv2d(max_pooled_1, w2, b2)

    max_pooled_2 = tf.squeeze(tf.nn.max_pool(conv_2, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding="VALID"), axis=(1, 2))
    min_pooled = tf.math.reduce_min(conv_2, axis=(1, 2))
    # concat_tensor = tf.concat([max_pooled_2, min_pooled], axis=1)

    fc_w1 = tf.Variable(tf.random_normal([80, 1024]))
    fc_b1 = tf.Variable(tf.random_normal([1024]))
    fc1 = tf.nn.relu(tf.add(tf.add(tf.matmul(max_pooled_2, fc_w1), fc_b1), tf.add(tf.matmul(min_pooled, fc_w1), fc_b1)))

    fc_w2 = tf.Variable(tf.random_normal([1024, 1024]))
    fc_b2 = tf.Variable(tf.random_normal([1024]))
    fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, fc_w2), fc_b2))

    fc_w3 = tf.Variable(tf.random_normal([1024, 1]))
    fc_b3 = tf.Variable(tf.random_normal([1]))
    fc3 = tf.squeeze(tf.nn.relu(tf.add(tf.matmul(fc2, fc_w3), fc_b3)), axis=1)

    loss = tf.reduce_min(tf.math.abs(tf.math.subtract(fc3, label_placeholder)))
    optimiser = tf.train.AdamOptimizer(learning_rate=LR)
    train_op = optimiser.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    training_image_paths, training_eval_paths, validation_image_paths, validation_eval_paths, _, _ = get_datasets(first_part_path, second_part_path)
    training_patches, training_scores = create_feed_dict(training_image_paths, training_eval_paths)
    nr_of_training_examples = len(training_scores)
    nr_of_training_batches = math.ceil(nr_of_training_examples / batch_size)

    validation_patches, validation_scores = create_feed_dict(validation_image_paths, validation_eval_paths)
    nr_of_validation_examples = len(validation_scores)
    nr_of_validation_batches = math.ceil(nr_of_validation_examples / batch_size)
    val_losses = np.zeros((nr_of_validation_batches), dtype=np.float32)

    # random.seed(3796)
    patch_indices = list(range(nr_of_training_examples))
    for epoch_index in range(num_epoch):
        random.shuffle(patch_indices)
        training_patches = training_patches[np.array(patch_indices)]
        training_scores = training_scores[np.array(patch_indices)]
        for batch_index in range(nr_of_training_batches - 1):
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            loss_, _ = sess.run([loss, train_op], feed_dict={image_placeholder: training_patches[start_index:end_index], label_placeholder: training_scores[start_index:end_index]})
            print("Epoch:", epoch_index + 1, "Batch:", batch_index + 1, '/', nr_of_training_batches, 'Loss:', loss_)
        batch_index += 1
        start_index = batch_index * batch_size
        loss_, _ = sess.run([loss, train_op], feed_dict={image_placeholder: training_patches[start_index:], label_placeholder: training_scores[start_index:]})
        print("Epoch:", epoch_index + 1, "Batch:", batch_index + 1, '/', nr_of_training_batches, 'Loss:', loss_)

        for batch_index in range(nr_of_validation_batches - 1):
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            loss_ = sess.run(loss, feed_dict={image_placeholder: validation_patches[start_index:end_index], label_placeholder: validation_scores[start_index:end_index]})
            val_losses[batch_index] = loss_
            print("Validation Epoch:", epoch_index + 1, "Validation Batch:", batch_index + 1, 'Loss:', loss_)
        batch_index += 1
        start_index = batch_index * batch_size
        loss_, _ = sess.run([loss, train_op], feed_dict={image_placeholder: validation_patches[start_index:], label_placeholder: validation_scores[start_index:]})
        val_losses[batch_index] = loss_
        print("Validation Epoch:", epoch_index + 1, "Validation Batch:", batch_index + 1, 'Loss:', loss_)
        print(np.mean(val_losses))

if __name__ == '__main__':
    main()