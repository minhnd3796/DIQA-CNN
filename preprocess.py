import math
import cv2
import os
import random
import numpy as np
import utils

stride = 16

def get_score_from_eval(filename):
    with open(filename, 'rb') as infile:
        lines = [line for line in infile][:4]
    return float(lines[3].split()[0][:-1]) / 100

def get_image_and_eval_paths(set_paths):
    image_paths = []
    eval_paths = []
    for set_path in set_paths:
        files = os.listdir(set_path)
        for item in files:
            split_item = os.path.splitext(item)
            if split_item[1].lower() == '.jpg':
                image_paths.append(os.path.join(set_path, split_item[0] + split_item[1]))
                eval_paths.append(os.path.join(set_path, 'eval_' + split_item[0] + '.txt'))
    return image_paths, eval_paths

def get_datasets(first_part_path, second_part_path):
    first_sets = os.listdir(first_part_path)
    second_sets = os.listdir(second_part_path)
    set_paths = []
    for first_set in first_sets:
        if first_set[:3] == 'set':
            set_paths.append(os.path.join(first_part_path, first_set))
    for second_set in second_sets:
        if second_set[:3] == 'set':
            set_paths.append(os.path.join(second_part_path, second_set))

    num_sets = len(set_paths)
    indices = list(range(num_sets))
    random.seed(3796)
    random.shuffle(indices)

    shuffled_indices = np.array(indices)
    shuffled_set_paths = np.array(set_paths)[shuffled_indices]

    num_one_fold = num_sets // 5
    training_set_paths = shuffled_set_paths[:num_one_fold * 3]
    validation_set_paths = shuffled_set_paths[num_one_fold * 3:num_one_fold * 3 + num_one_fold]
    test_set_paths = shuffled_set_paths[num_one_fold * 4:]

    """ print('training_set_paths', (training_set_paths))
    print('validation_set_paths', (validation_set_paths))
    print('test_set_paths', (test_set_paths)) """

    training_image_paths, training_eval_paths = get_image_and_eval_paths(training_set_paths)
    validation_image_paths, validation_eval_paths = get_image_and_eval_paths(validation_set_paths)
    test_image_paths, test_eval_paths = get_image_and_eval_paths(test_set_paths)

    return training_image_paths, training_eval_paths, validation_image_paths, validation_eval_paths, test_image_paths, test_eval_paths

def locally_normalise_and_otsu_thresholding(grey_img):
    std = np.std(grey_img)
    mean = np.mean(grey_img)
    normalised_img = (grey_img - mean) / std

    th = cv2.adaptiveThreshold(cv2.medianBlur(grey_img, 31), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31,2)
    return normalised_img, th

def is_constant(patch):
    patch_uint32 = np.array(patch, dtype=np.uint32)
    if np.sum(patch_uint32) % np.size(patch) == 0:
        return True
    else:
        return False

def sift_patches(img, patch_size, stride):
    img_height, img_width = img.shape
    patch_index_list = []

    y = 0
    x = 0
    while y + stride < img_height:
        while x + stride < img_width:
            patch_index_list.append((y, x))
            x += stride
        y += stride
    y = 0
    while y + stride < img_height:
        patch_index_list.append((y, img_width - patch_size))
        y += stride
    x = 0
    while x + stride < img_width:
        patch_index_list.append((img_height - patch_size, x))
        x += stride
    patch_index_list.append((img_height - patch_size, img_width - patch_size))

    """ num_row_patches = math.ceil(img_height / patch_size)
    num_col_patches = math.ceil(img_width / patch_size)
    for row_patch_index in range(0, num_row_patches - 1):
        for col_patch_index in range(0, num_col_patches - 1):
            y = patch_size * row_patch_index
            x = patch_size * col_patch_index
            patch_index_list.append((y, x))
    last_y = img_height - patch_size
    for col_patch_index in range(0, num_col_patches - 1):
        x = patch_size * col_patch_index
        patch_index_list.append((last_y, x))
    last_x = img_width - patch_size
    for row_patch_index in range(0, num_row_patches - 1):
        y = patch_size * row_patch_index
        patch_index_list.append((y, last_x))
    patch_index_list.append((last_y, last_x)) """
    return patch_index_list

def create_feed_dict(image_paths, eval_paths):
    patch_size = 48
    patches = []
    scores = []
    for i in range(len(image_paths)):
        grey_img = cv2.imread(image_paths[i], 0)
        corners = utils.get_document_corners_from_grey(grey_img)
        grey_img = utils.four_point_transform(grey_img, corners)
        normalised_img, otsu_thresh = locally_normalise_and_otsu_thresholding(grey_img)
        normalised_img = np.expand_dims(normalised_img, axis=2)
        patch_indices = sift_patches(grey_img, patch_size, stride)
        for (y, x) in patch_indices:
            if not is_constant(otsu_thresh[y:y + patch_size, x:x + patch_size]):
                patches.append(normalised_img[y:y + patch_size, x:x + patch_size, :])
                score = get_score_from_eval(eval_paths[i])
                scores.append(score)
    return patches, np.array(scores, dtype=np.float32)

def create_eval_feed_dict(image_paths, eval_paths):
    patch_size = 48
    patches = []
    scores = []
    for i in range(len(image_paths)):
        patches_of_one_image = []
        score = get_score_from_eval(eval_paths[i])
        scores.append(score)
        grey_img = cv2.imread(image_paths[i], 0)
        normalised_img, otsu_thresh = locally_normalise_and_otsu_thresholding(grey_img)
        normalised_img = np.expand_dims(normalised_img, axis=2)
        patch_indices = sift_patches(grey_img, patch_size, stride)
        for (y, x) in patch_indices:
            if not is_constant(otsu_thresh[y:y + patch_size, x:x + patch_size]):
                patches_of_one_image.append(normalised_img[y:y + patch_size, x:x + patch_size, :])
        patches.append(np.array(patches_of_one_image, dtype=np.float32))
    return patches, np.array(scores, dtype=np.float32)

""" first_part_path = '../DIQA_Release_1.0_Part1'
second_part_path = '../DIQA_Release_1.0_Part2/FineReader/'
training_image_paths, training_eval_paths, validation_image_paths, validation_eval_paths, test_image_paths, test_eval_paths = get_datasets(first_part_path, second_part_path)

f = open('training.txt', 'w')
for path in training_image_paths:
    f.write(path[3:] + '\n')
f.close()
f = open('validation.txt', 'w')
for path in validation_image_paths:
    f.write(path[3:] + '\n')
f.close()
f = open('test.txt', 'w')
for path in test_image_paths:
    f.write(path[3:] + '\n')
f.close()
training_patches, training_scores = create_feed_dict(training_image_paths, training_eval_paths)
validation_patches, validation_scores = create_feed_dict(validation_image_paths, validation_eval_paths)
test_patches, test_scores = create_feed_dict(test_image_paths, test_eval_paths)

print('len(training_patches)', len(training_patches))
print('len(training_scores)', len(training_scores))
print('len(validation_patches)', len(validation_patches))
print('len(validation_scores)', len(validation_scores))
print('len(test_patches)', len(test_patches))
print('len(test_scores)', len(test_scores)) """