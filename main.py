import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import argparse
from data_loader.data_utils import *
from models.trainer import *


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument('--image_w', type=int, default=75)
parser.add_argument('--image_h', type=int, default=75)
parser.add_argument('--n_class', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--K_fold', type=int, default=4)
parser.add_argument('--test_size', type=float, default=0.33)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--keep_prob', type=float, default=0.8)
# K_fold cross_validation or not
parser.add_argument('--split', type=bool, default=False)
# subject_independent or subject_dependent
parser.add_argument('--independent', type=bool, default=False)

args = parser.parse_args()
print(f'Training configs:{args}')


fold_data = load_cv_data('data_loader/gamma_de_o_all2', args)
    
if __name__ == '__main__':
    model_train_and_val(fold_data, args)
