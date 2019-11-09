import tensorflow as tf
from data_loader.data_utils import *
from models.base_model import *
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from utils.save_utils import *


def model_train_and_val(fold_data, args):
    """
    Train the model and validate it.
    :param fold_data:instance of DataSet, data source for training.
    :param args:instance of argparse, arguments for training.
    """
    fold_all = []
    best_fold_acc = 0.0
    best_fold = {}
    if args.split:
        fold_num = args.K_fold
    else:
        fold_num = len(fold_data.val_data_file)

    for i in range(fold_num):
        print("========== [Fold-%d] ==========" % i)

        tf.reset_default_graph()

        train_files, train_labels = fold_data.train_data_file[i], fold_data.train_data_label[i]
        val_files, val_labels = fold_data.val_data_file[i], fold_data.val_data_label[i]

        train_dataset = get_batch(train_files, train_labels, args.batch_size)
        val_dataset = get_batch(val_files, val_labels, args.batch_size)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes)

        data, label = iterator.get_next()
        train_iterator = train_dataset.make_initializable_iterator()
        val_iterator = val_dataset.make_initializable_iterator()

        fold = {'fold': i, 'best_epoch': 0, 'best_acc': 0.0, 'best_pre': 0.0, 'best_recall': 0.0,
                'all_train_loss': np.zeros(args.epoch), 'all_train_acc': np.zeros(args.epoch),
                'all_valid_loss': np.zeros(args.epoch), 'all_valid_acc': np.zeros(args.epoch),
                'all_valid_pre': np.zeros(args.epoch), 'all_valid_recall': np.zeros(args.epoch)}

        with tf.Session() as sess:
            training_handle = sess.run(train_iterator.string_handle())
            val_handel = sess.run(val_iterator.string_handle())

            x = data
            y = label

            dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            global_step = tf.Variable(0, name="global_step", trainable=False)

            logits = build_model(x, args)
            loss = losses(logits, y)
            train_op = train(loss, args.learning_rate, global_step)
            acc, true = evaluation(logits, y)
            pre = prediction(logits)
            print(" Training Model initialized")

            # saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
            sess.run(tf.global_variables_initializer())

            for epoch in np.arange(args.epoch):
                print("Training:Epoch number: {}".format(epoch))

                train_step = 1
                train_y = []
                train_y_true = []
                train_total_loss, train_n_batches = 0.0, 0
                sess.run(train_iterator.initializer)
                try:
                    while True:
                        _, _, train_loss, train_acc, train_pre, train_true = sess.run(
                            [train_op, global_step, loss, acc, pre, true],
                            feed_dict={handle: training_handle, dropout_keep_prob: args.keep_prob})
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

                train_accuary = np.mean(train_total_y_pred == train_total_y_true)

                print("Validating:Epoch number: {}".format(epoch))
                val_step = 1
                val_y = []
                val_y_true = []
                val_total_loss, val_n_batches = 0.0, 0
                sess.run(val_iterator.initializer)
                try:
                    while True:
                        _, val_loss, val_pre, val_score, val_acc, val_true = sess.run(
                            [global_step, loss, pre, logits, acc, true],
                            feed_dict={handle: val_handel, dropout_keep_prob: 1})
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
                
                val_matrix = confusion_matrix(val_total_y_true, val_total_y_pred)
                val_accuary = np.mean(val_total_y_pred == val_total_y_true)
                val_pre = precision_score(val_total_y_true, val_total_y_pred, average='macro')
                val_recall = recall_score(val_total_y_true, val_total_y_pred, average='macro')

                fold['all_train_loss'][epoch] = train_total_loss
                fold['all_train_acc'][epoch] = train_accuary

                fold['all_valid_loss'][epoch] = val_total_loss
                fold['all_valid_acc'][epoch] = val_accuary
                fold['all_valid_pre'][epoch] = val_pre
                fold['all_valid_recall'][epoch] = val_recall

                if val_accuary > fold['best_acc']:
                    fold['best_acc'] = val_accuary
                    fold['best_epoch'] = epoch
                    fold['best_pre'] = val_pre
                    fold['best_recall'] = val_recall
                    fold['best_matrix'] = val_matrix
                    # if val_accuary > best_fold_acc:
                    #     best_fold_acc = val_accuary
                    #     model_save(sess, global_step, 'CNN')
            if fold['best_acc'] > best_fold_acc:
                best_fold = fold
            print('fold:{},best_epoch:{},best_acc:{:.3},best_pre:{:.3},best_recall:{:.3}'.format(
                fold['fold'], fold['best_epoch'], fold['best_acc'], fold['best_pre'], fold['best_recall']))
            fold_all.append(fold)
            save_curve(fold)
        sess.close()
    save_confusion_matrix(best_fold['best_matrix'])
    summary_print(fold_all)
