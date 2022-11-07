import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(42)

# files location
dir_path = os.path.dirname(os.path.realpath(__file__))
TRAIN_FEAT = os.path.join(dir_path, 'data/train.csv')
TRAIN_LABEL = os.path.join(dir_path, 'data/train_result.csv')
TEST_FEAT = os.path.join(dir_path, 'data/test.csv')

NUM_CLASS = 19

train_data = pd.read_csv(TRAIN_FEAT)
train_label = pd.read_csv(TRAIN_LABEL)
test_data = pd.read_csv(TEST_FEAT)

def plot_topn_samples(top_n=2):
    feat = train_data.to_numpy()[:top_n, :-1]
    feat = feat.reshape(-1, 28, 56)
    label = train_label.to_numpy()[:top_n]
    fig = plt.figure()
    for i in range (feat.shape[0]):
        thisfig = fig.add_subplot(1, 2, i+1)
        thisfig.imshow(feat[i, ...], cmap = "gray")
        thisfig.set_title(label[i, 1])
    plt.show()

def preprocess_train_dataset():
    feat = train_data.to_numpy()[..., :-1]
    label = train_label.to_numpy()
    onehot_label = label_convert_to_onehot(label)
    feat_all = np.concatenate((feat, onehot_label), axis = -1)
    return feat_all

def preprocess_test_dataset():
    feat = test_data.to_numpy()[..., :-1]
    return feat

def split_dataset():
    feat_all = preprocess_train_dataset()
    np.random.shuffle(feat_all)
    train = feat_all[0:45000, ...]
    valid = feat_all[45000:50000, ...]
    return train, valid

def label_convert_to_onehot(label):
    '''Convert label in numpy to onehot array
    '''
    onehot = np.zeros((label.shape[0],19)) # 19 is number of classes
    onehot[np.arange(label.shape[0]), label[..., -1]] = 1
    return onehot

if __name__ == '__main__':
    plot_topn_samples(2)