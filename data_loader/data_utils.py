import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut


class DataSet(object):
    def __init__(self, fold_train, fold_val):
        self.train_data_file = fold_train['train_file']
        self.train_data_label = fold_train['train_label']
        self.val_data_file = fold_val['val_file']
        self.val_data_label = fold_val['val_label']


def load_cv_data(file_dir, args):
    """
    generate data in the form of DataSet.
    :param file_dir: str, the file path of data.
    :param args: instance of class argparse, arguments for training.
    :return: instance of class DataSet, the data source for training and validation.
    """
    fold_train_file = []
    fold_train_label = []
    fold_val_file = []
    fold_val_label = []

    if args.split:
        files = []
        label = []
        for file_name in os.listdir(file_dir):
            files.append(file_dir + '/' + file_name)
            label.append(int(file_name[0]))
        # fold_files, fold_labels, X_test, y_test = split_data(file_dir, args)
        skf = StratifiedKFold(n_splits=args.K_fold, random_state=0, shuffle=True)
        files = np.array(files)
        label = np.array(label)
        for train_index, val_index in skf.split(files, label):
            train_file, train_label = files[train_index], label[train_index]
            val_file, val_label = files[val_index], label[val_index]
            fold_train_file.append(train_file)
            fold_train_label.append(train_label)
            fold_val_file.append(val_file)
            fold_val_label.append(val_label)
    elif args.independent:
        subject_files = []
        for file_name in os.listdir(file_dir):
            subject_files.append(file_dir + '/' + file_name)
        loo = LeaveOneOut()

        for train_subject_index, val_subject_index in loo.split(subject_files):
            train_files = []
            train_labels = []
            val_files = []
            val_labels = []
            for index in train_subject_index:
                for file_name in os.listdir(subject_files[index]):
                    train_files.append(subject_files[index] + '/' + file_name)
                    train_labels.append(int(file_name[0]))
            for index in val_subject_index:
                for file_name in os.listdir(subject_files[index]):
                    val_files.append(subject_files[index] + '/' + file_name)
                    val_labels.append(int(file_name[0]))
            fold_train_file.append(train_files)
            fold_train_label.append(train_labels)
            fold_val_file.append(val_files)
            fold_val_label.append(val_labels)
    else:
        #test = ['7', '10', '8', '11', '9', '12']
        #test = ['7', '4', '8', '5', '9', '6']
        test = ['10', '11', '12', '13', '14', '15']
        #test = ['1', '2', '3', '13', '14', '15']
        #test = ['1', '2', '3', '5', '4', '6']
        train_files = []
        train_labels = []
        val_files = []
        val_labels = []
        for file_name in os.listdir(file_dir):
            if file_name.split('_')[2] in test:
                for file in os.listdir(file_dir + '/' + file_name):
                    val_files.append(file_dir + '/' + file_name + '/' + file)
                    val_labels.append(int(file[0]))
            else:
                for file in os.listdir(file_dir + '/' + file_name):
                    train_files.append(file_dir + '/' + file_name + '/' + file)
                    train_labels.append(int(file[0]))
        fold_train_file.append(train_files)
        fold_train_label.append(train_labels)
        fold_val_file.append(val_files)
        fold_val_label.append(val_labels)

    fold_train = {'train_file': fold_train_file, 'train_label': fold_train_label}
    fold_val = {'val_file': fold_val_file, 'val_label': fold_val_label}
    fold_data = DataSet(fold_train, fold_val)
    return fold_data


def _parse_function(filename, label):
    """
    :param filename: np.array, the list of image file names.
    :param label: np.array, the list of image labels.
    :return: Tensor, [image_w, image_h, channels]; Tensor, [label]
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    image_resized = tf.image.resize_images(image, [75, 75])
    return image_resized, label


def get_batch(images, label, batch_size):
    """
    generate data iteration batch.
    :param images:list, the list of image file names.
    :param label:list, the list of image labels.
    :param batch_size:int, the batch_size of training.
    :return: Dataset, dataset for iteration
    """
    images = np.array(images)
    label = np.array(label)
    dataset = tf.data.Dataset.from_tensor_slices((images, label))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=100000).batch(batch_size, drop_remainder=True)
    return dataset
