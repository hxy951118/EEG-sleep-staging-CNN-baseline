import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import tensorflow as tf


def save_confusion_matrix(cm):
    """
    plot and save the figure of confusion_matrix
    :param cm:
    :return:
    """
    cm_normalized = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3)
    plt.matshow(cm_normalized, cmap=plt.cm.Greens)
    for x in range(len(cm_normalized)):
        for y in range(len(cm_normalized)):
            plt.annotate(cm_normalized[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    num_local = np.array(range(len(cm)))
    lables_name = ['positive', 'neutral', 'negative']
    plt.xticks(num_local, lables_name)
    plt.yticks(num_local, lables_name)
    plt.savefig('./output/save_fig/confusion_matrix.png')


def save_curve(fold):
    plt.figure(figsize=(10, 7))
    plt.subplot(211)
    plt.plot(fold['all_valid_loss'], label='loss')
    plt.ylabel('loss')
    plt.title('Validation Loss')
    plt.legend(loc=0)

    plt.subplot(212)
    plt.plot(fold['all_valid_acc'], label='acc')
    plt.ylabel('accuracy')
    plt.title('Validation accuracy')
    plt.legend(loc=0)

    plt.savefig('./output/save_fig/fold_{}_valid.png'.format(fold['fold']))

    plt.figure(figsize=(10, 7))
    plt.subplot(211)
    plt.plot(fold['all_train_loss'], label='loss')
    plt.ylabel('loss')
    plt.title('Train Loss')
    plt.legend(loc=0)

    plt.subplot(212)
    plt.plot(fold['all_train_acc'], label='acc')
    plt.ylabel('accuracy')
    plt.title('Train accuracy')
    plt.legend(loc=0)

    plt.savefig('./output/save_fig/fold_{}_train.png'.format(fold['fold']))


def summary_print(fold_all):
    epoch = fold_all[0]['best_epoch']
    avg_valid_acc = np.mean(np.array([fold['all_valid_acc'][epoch] for fold in fold_all]))
    avg_valid_pre = np.mean(np.array([fold['all_valid_pre'][epoch] for fold in fold_all]))
    avg_valid_recall = np.mean(np.array([fold['all_valid_recall'][epoch] for fold in fold_all]))
    all_valid_std = np.std(np.array([fold['all_valid_acc'][epoch] for fold in fold_all]), ddof=1)
    print('Summary:')
    for fold in fold_all:
        print('fold:{},best_epoch:{},best_acc:{:.3},best_pre:{:.3},best_recall:{:.3}'.format(
            fold['fold'], fold['best_epoch'], fold['best_acc'], fold['best_pre'], fold['best_recall']))
    print('avg_acc:{:.3},avg_pre:{:.3},avg_recall:{:.3},all_std:{:.3}'.format(avg_valid_acc, avg_valid_pre,
                                                                              avg_valid_recall, all_valid_std))


def model_save(sess, global_steps, model_name, save_path='./output/models/'):
    """
    Save the checkpoint of trained model.
    :param sess: tf.Session().
    :param global_steps: tensor, record the global step of training in epochs.
    :param model_name: str, the name of saved model.
    :param save_path: str, the path of saved model.
    """
    saver = tf.train.Saver(max_to_keep=1)
    prefix_path = saver.save(sess, join(save_path, model_name), global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...')
