import os
import tensorflow as tf
import numpy as np
import config_train
from sklearn.model_selection import StratifiedKFold
Config = config_train.Config()


class DataLoader(object):
    def __int__(self, data_dir, n_folds, fold_idx):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx


    def load_cv_data(self, file_dir):
        file_dir = self.data_dir
        files = []
        label = []
        for file in os.listdir(file_dir):
            if file[0] == 'A':
                for sub_file in os.listdir(self.data_dir + file):
                    files.append(self.data_dir + file + '/' +sub_file)
                    label.append(0)
            elif file[0] == 'B':
                for sub_file in os.listdir(self.data_dir + file):
                    files.append(self.data_dir + file + '/' +sub_file)
                    label.append(1)
            elif file[0] == 'C':
                for sub_file in os.listdir(self.data_dir + file):
                    files.append(self.data_dir + file + '/' +sub_file)
                    label.append(2)
        skf = StratifiedKFold(n_splits=4, random_state=0, shuffle=True)
        files = np.array(files)
        label = np.array(label)
        fold_train_file = []
        fold_train_label = []
        fold_val_file = []
        fold_val_label = []
        for train_index, val_index in skf.split(files, label):
            train_file, train_label = files[train_index], label[train_index]
            val_file, val_label = files[val_index], label[val_index]
            fold_train_file.append(train_file)
            fold_train_label.append(train_label)
            fold_val_file.append(val_file)
            fold_val_label.append(val_label)
        return fold_train_file, fold_train_label, fold_val_file, fold_val_label

    def _parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=1)
        image = tf.cast(image_decoded, tf.float32)
        image_resized = tf.image.resize_images(image, [20, 20])
        return image_resized, label

    def get_batch(self, images, label):
        images = np.array(images)
        label = np.array(label)
        print(images)
        dataset = tf.data.Dataset.from_tensor_slices((images, label))
        dataset = dataset.map(self._parse_function)
        dataset = dataset.shuffle(buffer_size=100000).batch(Config.batch_size, drop_remainder=True)
        return dataset




if __name__ == '__main__':
    data_dir = 'data_split/'

    data_loader = DataLoader()
    data_loader.data_dir = data_dir
    data_loader.n_folds = 4
    data_loader.fold_idx = 0

    fold_train_file, fold_train_label, fold_val_file, fold_val_label = data_loader.load_cv_data(data_dir)

    for i in range(4):
        train_files, train_lables = fold_train_file[i], fold_train_label[i]
        val_files, val_labels = fold_val_file[i], fold_val_label[i]
        

        train_dataset = data_loader.get_batch(train_files, train_lables)
        val_dataset = data_loader.get_batch(val_files, val_labels)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes)

        data, label = iterator.get_next()
        train_iterator = train_dataset.make_initializable_iterator()
        val_iterator = val_dataset.make_initializable_iterator()

        with tf.Session() as sess:
            training_handle = sess.run(train_iterator.string_handle())
            val_handel = sess.run(val_iterator.string_handle())

            x = data
            y = label

            for j in range(2):
                sess.run(train_iterator.initializer)
                i =0
                print('train:')
                try:
                    while True:
                        print(sess.run([y],  feed_dict={handle: training_handle}))
                        i += 1
                        
                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')

                i = 0
                print('test:')

                sess.run(val_iterator.initializer)
                try:
                    while True:
                        print(sess.run([y], feed_dict={handle: val_handel}))
                        i += 1
                        
                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')

    # train_files, val_files = data_loader.load_cv_data(os.listdir(data_dir))
    # print(train_files)
    # print(val_files)
    #
    # train_dataset = data_loader.get_train_batch(train_files)
    # val_dataset = data_loader.get_train_batch(val_files)
    # print(type(train_dataset))
    #
    # handle = tf.placeholder(tf.string, shape=[])
    # iterator = tf.data.Iterator.from_string_handle(
    #     handle, train_dataset.output_types, train_dataset.output_shapes)
    #
    # data, label = iterator.get_next()
    # train_iterator = train_dataset.make_initializable_iterator()
    # val_iterator = val_dataset.make_initializable_iterator()
    #
    # with tf.Session() as sess:
    #     training_handle = sess.run(train_iterator.string_handle())
    #     val_handel = sess.run(val_iterator.string_handle())
    #
    #     x = data
    #     y = label
    #
    #     for j in range(2):
    #         sess.run(train_iterator.initializer)
    #         i =0
    #         print('train:')
    #         try:
    #             while True:
    #                 sess.run([x, y],  feed_dict={handle: training_handle})
    #                 i += 1
    #                 print(i)
    #         except tf.errors.OutOfRangeError:
    #             print('Done training -- epoch limit reached')
    #
    #         i = 0
    #         print('test:')
    #
    #         sess.run(val_iterator.initializer)
    #         try:
    #             while True:
    #                 sess.run([x, y], feed_dict={handle: val_handel})
    #                 i += 1
    #                 print(i)
    #         except tf.errors.OutOfRangeError:
    #             print('Done training -- epoch limit reached')


















# def get_batch(data, label):
#     data = tf.data.Dataset.from_tensor_slices((data, label))
#     data = data.shuffle(buffer_size=1000).batch(BATCH_SIZE, drop_remainder=True).repeat()
#     iterator = data.make_one_shot_iterator()
#     # data, label = iterator.get_next()
#     # label = tf.reshape(label, [BATCH_SIZE])
#     # data = tf.cast(data, tf.float32)
#     return iterator
