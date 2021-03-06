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
num_epoch = 10000
batch_size = 512

def conv(current, number, in_channels, out_channels, init):
    filters = tf.get_variable(name='conv' + str(number) + '_' + 'W',
                              initializer=init, shape=(5, 5, in_channels, out_channels))
    bias = tf.get_variable(name='conv' + str(number) + '_' + 'b',
                           initializer=init, shape=(out_channels))
    current = tf.nn.bias_add(tf.nn.conv2d(current, filters, strides=[1, 1, 1, 1], padding="VALID"), bias)
    return current

def forward_validation(image_patches, batch_size, sess, fc3, image_placeholder, keep_prob, loss, score, label_placeholder):
    nr_of_examples = len(image_patches)
    patch_scores = np.ones((nr_of_examples,), dtype=np.float32) * score
    nr_of_batches = math.ceil(nr_of_examples / batch_size)
    patch_scores = np.zeros(nr_of_examples)
    batch_index = -1
    f = open('logs/val_loss.txt', 'a')
    for batch_index in range(nr_of_batches - 1):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        fc3_, loss_ = sess.run([fc3, loss], feed_dict={image_placeholder: image_patches[start_index:end_index], keep_prob: 1., label_placeholder: patch_scores[start_index:end_index]})
        f.write(str(loss_) + '\n')
        patch_scores[start_index:end_index] = fc3_
    batch_index += 1
    start_index = batch_index * batch_size
    fc3_, loss_ = sess.run([fc3, loss], feed_dict={image_placeholder: image_patches[start_index:], keep_prob: 1., label_placeholder: patch_scores[start_index:]})
    f.write(str(loss_) + '\n')
    f.close()
    patch_scores[start_index:] = fc3_
    return np.mean(patch_scores)

def forward_training(image_patches, btch_size, sess, fc3, image_placeholder, keep_prob, loss, score, label_placeholder):
    nr_of_examples = len(image_patches)
    patch_scores = np.ones((nr_of_examples,), dtype=np.float32) * score
    nr_of_batches = math.ceil(nr_of_examples / batch_size)
    patch_scores = np.zeros(nr_of_examples)
    batch_index = -1
    f = open('logs/training_loss.txt', 'a')
    for batch_index in range(nr_of_batches - 1):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        fc3_, loss_ = sess.run([fc3, loss], feed_dict={image_placeholder: image_patches[start_index:end_index], keep_prob: 1., label_placeholder: patch_scores[start_index:end_index]})
        f.write(str(loss_) + '\n')
        patch_scores[start_index:end_index] = fc3_
    batch_index += 1
    start_index = batch_index * batch_size
    fc3_, loss_ = sess.run([fc3, loss], feed_dict={image_placeholder: image_patches[start_index:], keep_prob: 1., label_placeholder: patch_scores[start_index:]})
    f.write(str(loss_) + '\n')
    f.close()
    patch_scores[start_index:] = fc3_
    return np.mean(patch_scores)

def main():
    LR = 0.001
    learning_rate_decay_epochs = 20
    image_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 1), name='image_placeholder')
    label_placeholder = tf.placeholder(dtype=tf.float32, shape=(None), name='label_placeholder')
    learning_rate_placeholder = tf.placeholder(dtype=tf.float32, name='learning_rate_placeholder')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    
    learning_rate_decay_factor = 0.95
    
    init = tf.contrib.layers.xavier_initializer()

    conv_1 = conv(image_placeholder, 1, 1, 40, init)

    max_pooled_1 = tf.nn.max_pool(conv_1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="VALID")

    conv_2 = conv(max_pooled_1, 2, 40, 80, init)

    max_pooled_2 = tf.squeeze(tf.nn.max_pool(conv_2, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding="VALID"), axis=(1, 2))
    min_pooled = tf.reduce_min(conv_2, axis=(1, 2))

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

    loss = tf.reduce_mean(tf.abs(tf.subtract(fc3, label_placeholder)), name='loss')
    tf.summary.scalar('training_loss', loss)

    optimiser = tf.train.GradientDescentOptimizer(learning_rate=LR)
    global_step = tf.Variable(0, trainable=False)
    train_op = optimiser.minimize(loss, global_step=global_step)

    training_image_paths, training_eval_paths, validation_image_paths, validation_eval_paths, test_image_paths, test_eval_paths = get_datasets(first_part_path, second_part_path)
    training_patches, training_scores = create_feed_dict(training_image_paths, training_eval_paths)
    
    nr_of_training_examples = len(training_scores)
    nr_of_training_batches = math.ceil(nr_of_training_examples / batch_size)

    learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step, learning_rate_decay_epochs * nr_of_training_batches, learning_rate_decay_factor, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    eval_training_patches, eval_training_scores = create_eval_feed_dict(training_image_paths, training_eval_paths)
    validation_patches, validation_scores = create_eval_feed_dict(validation_image_paths, validation_eval_paths)
    # test_patches, test_scores = create_eval_feed_dict(test_image_paths, test_eval_paths)

    # Create a saver
    saver = tf.train.Saver(tf.trainable_variables())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    log_dir = 'logs'
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    # random.seed(3796)
    patch_indices = list(range(nr_of_training_examples))
    for epoch_index in range(num_epoch):
        random.shuffle(patch_indices)
        training_patches = training_patches[np.array(patch_indices)]
        training_scores = training_scores[np.array(patch_indices)]
        for batch_index in range(nr_of_training_batches - 1):
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            loss_, _, summary_str, step_ = sess.run([loss, train_op, summary_op, global_step], feed_dict={learning_rate_placeholder: LR, image_placeholder: training_patches[start_index:end_index], label_placeholder: training_scores[start_index:end_index], keep_prob: 1.})
            print('Step:', step_, "Epoch:", epoch_index + 1, "Batch:", batch_index + 1, '/', nr_of_training_batches, 'Loss:', loss_)
            summary_writer.add_summary(summary_str, global_step=step_)
        batch_index += 1
        start_index = batch_index * batch_size
        loss_, _, summary_str, step_ = sess.run([loss, train_op, summary_op, global_step], feed_dict={learning_rate_placeholder: LR, image_placeholder: training_patches[start_index:], label_placeholder: training_scores[start_index:], keep_prob: 1.})
        print('Step:', step_, "Epoch:", epoch_index + 1, "Batch:", batch_index + 1, '/', nr_of_training_batches, 'Loss:', loss_)
        summary_writer.add_summary(summary_str, global_step=step_)

        predicted_training_scores = np.zeros_like(eval_training_scores)
        
        summary = tf.Summary()
        for i in range(len(eval_training_patches)):
            predicted_training_scores[i] = forward_training(eval_training_patches[i], batch_size, sess, fc3, image_placeholder, keep_prob, loss, eval_training_scores[i], label_placeholder)
        training_lcc = pearsonr(predicted_training_scores, eval_training_scores)[0]
        training_srocc = spearmanr(predicted_training_scores, eval_training_scores)[0]
        print("Training LCC:", training_lcc)
        print("Training SROCC:", training_srocc)

        predicted_validation_scores = np.zeros_like(validation_scores)
        for i in range(len(validation_patches)):
            predicted_validation_scores[i] = forward_validation(validation_patches[i], batch_size, sess, fc3, image_placeholder, keep_prob, loss, validation_scores[i], label_placeholder)
        validation_lcc = pearsonr(predicted_validation_scores, validation_scores)[0]
        validation_srocc = spearmanr(predicted_validation_scores, validation_scores)[0]
        print("Validation LCC:", validation_lcc)
        print("Validation SROCC:", validation_srocc)

        summary.value.add(tag='training_lcc', simple_value=training_lcc)
        summary.value.add(tag='training_srocc', simple_value=training_srocc)
        summary.value.add(tag='validation_lcc', simple_value=validation_lcc)
        summary.value.add(tag='validation_srocc', simple_value=validation_srocc)
        summary_writer.add_summary(summary, global_step=step_)

        saver.save(sess, log_dir + '/model.ckpt', global_step=step_)        

    """ predicted_test_scores = np.zeros_like(test_scores)
    for i in range(len(test_patches)):
        predicted_test_scores[i] = forward(test_patches[i], batch_size, sess, fc3, image_placeholder, keep_prob, loss, global_step)
    print("Test LCC:", pearsonr(predicted_test_scores, test_scores)[0])
    print("Test SROCC:", spearmanr(predicted_test_scores, test_scores)[0]) """ 

if __name__ == '__main__':
    main()
