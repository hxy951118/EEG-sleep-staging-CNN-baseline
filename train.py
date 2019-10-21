import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import tensorflow as tf
from data_loader import DataLoader
from model import model
from config_train import Config
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

model = model()
config = Config()
def train(n_flods, fold_idx):

    data_dir = 'e2_jl/'

    data_loader = DataLoader()
    data_loader.data_dir = data_dir
    data_loader.n_folds = n_flods
    data_loader.fold_idx = fold_idx

    fold_train_file, fold_train_label, fold_val_file, fold_val_label = data_loader.load_cv_data(data_dir)

    for i in range(4):
        tf.reset_default_graph()
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


        best_fscore = 0.0
        best_acc = 0.0
        best_kappa = 0.0
        #min_fpr = 999


        with tf.Session() as sess:
            # model = model()
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True)
            sess = tf.Session(config=session_conf)

            training_handle = sess.run(train_iterator.string_handle())
            val_handel = sess.run(val_iterator.string_handle())

            x = data
            y = label

            dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            global_step = tf.Variable(0, name="global_step", trainable=False)

            logits = model.inference(x, config.batch_size, config.n_class, dropout_keep_prob)
            loss = model.losses(logits, y)
            train_op = model.trainning(loss, config.learning_rate, global_step)
            acc, true = model.evaluation(logits, y)
            pre = model.prediction(logits)
            print(" Training Model initialized")

            saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
            sess.run(tf.global_variables_initializer())

            n_epochs = config.training_epoch
            all_train_loss = np.zeros(n_epochs)
            all_train_acc = np.zeros(n_epochs)
            all_train_f1 = np.zeros(n_epochs)
            all_valid_loss = np.zeros(n_epochs)
            all_valid_acc = np.zeros(n_epochs)
            all_valid_pre = np.zeros(n_epochs)
            all_valid_recall = np.zeros(n_epochs)
            #all_valid_fpr = np.zeros(n_epochs)


            for epoch in np.arange(config.training_epoch):
                print("Epoch number: {}".format(epoch))

                train_step = 1
                train_y = []
                train_y_true = []
                train_total_loss, train_n_batches = 0.0, 0
                sess.run(train_iterator.initializer)
                try:
                    while True:
                        _, _, train_loss, train_acc, train_pre, train_true = sess.run(
                            [train_op, global_step, loss, acc, pre, true],
                            feed_dict={handle: training_handle, dropout_keep_prob: 0.5})
                        train_total_loss += train_loss
                        train_n_batches += 1
                        train_y.append(train_pre)
                        train_y_true.append(train_true)
                        print(" Step %d, train loss = %.2f, train accuracy = %.2f%%" % (
                                train_step, train_loss, train_acc * 100))
                        train_step += 1
                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch{}'.format(epoch))
                train_total_loss /= train_n_batches
                train_total_y_true = np.hstack(train_y_true)
                train_total_y_pred = np.hstack(train_y)

                # train_cm = confusion_matrix(train_total_y_true, train_total_y_pred)
                train_accuary = np.mean(train_total_y_pred == train_total_y_true)
                train_f1 = f1_score(train_total_y_true, train_total_y_pred, average="macro")
                #FP = 0
                #N = 0 
                #for index in range(len(train_total_y_true)):
                    #if (train_total_y_pred[index] == 1 and train_total_y_true[index] == 0):
                         # FP+=1
                    #if (train_total_y_true[index] ==0):
                     #   N+=1;
                #FPR = FP/N
                
                val_step = 1
                val_y = []
                val_y_true = []
                val_total_loss, val_n_batches = 0.0, 0
                sess.run(val_iterator.initializer)
                try:
                     while True:
                        _, val_loss, val_pre, val_score, val_acc, val_true = sess.run(
                            [global_step, loss, pre, logits, acc, true], feed_dict={handle: val_handel, dropout_keep_prob: 1})
                        val_total_loss += val_loss
                        val_n_batches += 1
                        val_y.append(val_pre)
                        val_y_true.append(val_true)
                        print("Step %d, val loss = %.2f, val accuracy = %.2f%%" % (val_step, val_loss, val_acc * 100))
                        val_step += 1
                except tf.errors.OutOfRangeError:
                    print('Done validation -- epoch{}'.format(epoch))
                val_total_loss /= val_n_batches
                val_total_y_true = np.hstack(val_y_true)
                val_total_y_pred = np.hstack(val_y)

                # val_cm = confusion_matrix(val_total_y_true, val_total_y_pred)
                val_accuary = np.mean(val_total_y_pred == val_total_y_true)
                val_pre = precision_score(val_total_y_true, val_total_y_pred,average='macro')
                val_recall = recall_score(val_total_y_true, val_total_y_pred,average='macro')

                all_train_loss[epoch] = train_total_loss
                all_train_acc[epoch] = train_accuary
                all_train_f1[epoch] = train_f1
                all_valid_loss[epoch] = val_total_loss
                all_valid_acc[epoch] = val_accuary
                all_valid_pre[epoch] = val_pre
                all_valid_recall[epoch] = val_recall
                #all_train_fpr[epoch] = FPR


                # save the best model
                if (val_accuary > best_acc):
                    best_acc = val_accuary
                
            print('last_val_accuarcy:{:.3},last_val_precision:{:.3},last_val_recall:{:.3}'.format(val_accuary, val_pre, val_recall))
            print('all_val_accuary:{:.3},all_val_precision:{:.3},all_val_recall:{:.3},besl_accuarcy:{:.3}'.format(np.mean(all_valid_acc), np.mean(all_valid_pre), np.mean(all_valid_recall), best_acc))
            
            for i in all_valid_acc:
                print(i)
           
            plt.figure(figsize=(12, 9))
            plt.subplot(311)
            plt.plot(all_valid_loss, label='loss')
            plt.ylabel('loss')
            plt.title('Validation Loss')
            plt.legend(loc=0)

            plt.subplot(312)
            plt.plot(all_valid_acc, label='acc')
            plt.ylabel('accuracy')
            plt.title('Validation accuracy')
            plt.legend(loc=0)

            plt.figure(figsize=(12, 9))
            plt.subplot(311)
            plt.plot(all_train_loss, label='loss')
            plt.ylabel('loss')
            plt.title('Train Loss')
            plt.legend(loc=0)

            plt.subplot(312)
            plt.plot(all_train_acc, label='acc')
            plt.ylabel('accuracy')
            plt.title('Train accuracy')
            plt.legend(loc=0)

            plt.subplot(313)
            plt.plot(all_train_f1, label='f1')
            plt.ylabel('f1')
            plt.title('Train f1')
            plt.legend(loc=0)
            plt.show()
        sess.close()


if __name__ == '__main__':
    train(4, 0)
