import math
import cv2
import os
import random
import numpy as np

def get_score_from_eval(filename):
    with open(filename, 'rb') as infile:
        lines = [line for line in infile][:4]
    return float(lines[3].split()[0][:-1]) / 100

def get_datasets(first_part_path, second_part_path):
    first_sets = os.listdir(first_part_path)
    image_paths = []
    eval_paths = [] 

    for doc in first_sets:
        files = os.listdir(os.path.join(first_part_path, doc))
        for item in files:
            split_item = os.path.splitext(item)
            if split_item[1].lower() == '.jpg':
                image_paths.append(os.path.join(first_part_path, doc, split_item[0] + split_item[1]))
                eval_paths.append(os.path.join(first_part_path, doc, 'eval_' + split_item[0] + '.txt'))
    second_sets = os.listdir(second_part_path)

    for doc in second_sets:
        if doc[:3] == 'set':
            files = os.listdir(os.path.join(second_part_path, doc))
            for item in files:
                split_item = os.path.splitext(item)
                if split_item[1].lower() == '.jpg':
                    image_paths.append(os.path.join(second_part_path, doc, split_item[0] + split_item[1]))
                    eval_paths.append(os.path.join(second_part_path, doc, 'eval_' + split_item[0] + '.txt'))

    num_docs = len(image_paths)
    indices = list(range(num_docs))
    # random.seed(3796)
    random.shuffle(indices)

    shuffled_indices = np.array(indices)
    shuffled_image_paths = np.array(image_paths)[shuffled_indices]
    shuffled_eval_paths = np.array(eval_paths)[shuffled_indices]

    num_one_fold = num_docs // 5
    training_image_paths = shuffled_image_paths[:num_one_fold * 3]
    training_eval_paths = shuffled_eval_paths[:num_one_fold * 3]

    validation_image_paths = shuffled_image_paths[num_one_fold * 3:num_one_fold * 3 + num_one_fold]
    validation_eval_paths = shuffled_eval_paths[num_one_fold * 3:num_one_fold * 3 + num_one_fold]

    test_image_paths = shuffled_image_paths[num_one_fold * 4:]
    test_eval_paths = shuffled_eval_paths[num_one_fold * 4:]

    return training_image_paths, training_eval_paths, validation_image_paths, validation_eval_paths, test_image_paths, test_eval_paths

def locally_normalise_and_otsu_thresholding(grey_img):
    std = np.std(grey_img)
    mean = np.mean(grey_img)
    normalised_img = (grey_img - mean) / std

    _, otsu_thresh = cv2.threshold(grey_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return normalised_img, otsu_thresh

def is_constant(patch):
    patch_uint32 = np.array(patch, dtype=np.uint32)
    if np.sum(patch_uint32) % np.size(patch) == 0:
        return True
    else:
        return False

def sift_patches(img, patch_size):
    img_height, img_width = img.shape
    num_row_patches = math.ceil(img_height / patch_size)
    num_col_patches = math.ceil(img_width / patch_size)
    patch_index_list = []

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
    patch_index_list.append((last_y, last_x))
    return patch_index_list

def create_feed_dict(image_paths, eval_paths):
    patch_size = 48
    patches = []
    scores = []
    for i in range(len(image_paths)):
        grey_img = cv2.imread(image_paths[i], 0)
        normalised_img, otsu_thresh = locally_normalise_and_otsu_thresholding(grey_img)
        normalised_img = np.expand_dims(normalised_img, axis=2)
        patch_indices = sift_patches(grey_img, patch_size)
        for (y, x) in patch_indices:
            if not is_constant(otsu_thresh[y:y + patch_size, x:x + patch_size]):
                patches.append(normalised_img[y:y + patch_size, x:x + patch_size, :])
                score = get_score_from_eval(eval_paths[i])
                scores.append(score)
    return np.array(patches, dtype=np.float32), np.array(scores, dtype=np.float32)

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
        patch_indices = sift_patches(grey_img, patch_size)
        for (y, x) in patch_indices:
            if not is_constant(otsu_thresh[y:y + patch_size, x:x + patch_size]):
                patches_of_one_image.append(normalised_img[y:y + patch_size, x:x + patch_size, :])
        patches.append(patches_of_one_image)
    return np.array(patches, dtype=np.float32), np.array(scores, dtype=np.float32)

first_part_path = '../DIQA_Release_1.0_Part1'
second_part_path = '../DIQA_Release_1.0_Part2/FineReader/'
training_image_paths, training_eval_paths, validation_image_paths, validation_eval_paths, test_image_paths, test_eval_paths = get_datasets(first_part_path, second_part_path)
training_patches, training_scores = create_eval_feed_dict(training_image_paths, training_eval_paths)
validation_patches, validation_scores = create_eval_feed_dict(validation_image_paths, validation_eval_paths)
test_patches, test_scores = create_eval_feed_dict(test_image_paths, test_eval_paths)

print(len(training_patches))
print(len(training_scores))
print(len(validation_patches))
print(len(validation_scores))
print(len(test_patches))
print(len(test_scores))
