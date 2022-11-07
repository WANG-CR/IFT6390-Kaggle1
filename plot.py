import numpy as np
import resnet
from resnet import Resnet
from logreg import LogisticRegression
from data_module import split_dataset
from tqdm import tqdm

import pandas as pd 
import pickle
import math
from train_model import train_model
import matplotlib.pyplot as plt

# # wd graph, when lr=1e-3, bs=200
# wd = [0, 1e-6, 1e-5, 1e-4]
# acc_wd = [0.2034, 0.2034, 0.1986, 0.1692 ]


# # bs graph, when lr=1e-3, wd=1e-6
# bs = [1, 200, 2000, 20000]
# acc_bs = [0.2202, 0.196, 0.1252, 0.0926]


# # lr graph, when wd = 1e-6, bs = 200
# lr = [0.0001, 0.001, 0.005, 0.01]
# acc_lr = [0.1254, 0.199, 0.2124, 0.2144]

def plot_graph(variable, model, x_axis, y_axis, log_scale=False):
    plt.plot(x_axis, y_axis, 'r-', marker="o")
    plt.ylabel("accuracy")
    plt.xlabel(variable)
    plt.title("Accuracy-" + variable + " curve for " + model + " model")
    if log_scale:
        plt.xscale('log')
    plt.show()


def logistic_regression_print_lr(fix_step = False):
    grid = {
        'lr_list':[1e-4, 1e-3, 5e-3, 1e-2],
        'wd_list':[1e-6],
        'bs_list':[200],
    }
    train_data, valid_data = split_dataset()
    acc = []
    for lr in grid['lr_list']:
        for wd in grid['wd_list']:
            for bs in tqdm(grid['bs_list']):
                print(f'<<<<<<<<<<<<<<<<<<')
                model = LogisticRegression(n_feat=56*28, n_class=19, lr=lr, weight_decay=wd)
                model, train_loss, train_acc, valid_acc = train_model(model, train_data, batch_size=bs, max_epoch=100, valid_data=valid_data, fix_step=fix_step)
                acc.append(valid_acc)
                print(f'lr: {lr} | wd : {wd} | bs: {bs}')
                print(f'valid accuracy: {valid_acc}')
    return grid['lr_list'], acc

def logistic_regression_print_wd(fix_step = False):
    grid = {
        'lr_list':[1e-3],
        'wd_list':[0, 1e-6, 1e-5, 1e-4],
        'bs_list':[200],
    }
    acc = []
    train_data, valid_data = split_dataset()
    for lr in grid['lr_list']:
        for wd in grid['wd_list']:
            for bs in tqdm(grid['bs_list']):
                print(f'<<<<<<<<<<<<<<<<<<')
                model = LogisticRegression(n_feat=56*28, n_class=19, lr=lr, weight_decay=wd)
                model, train_loss, train_acc, valid_acc = train_model(model, train_data, batch_size=bs, max_epoch=100, valid_data=valid_data,fix_step=fix_step)
                acc.append(valid_acc)
                print(f'lr: {lr} | wd : {wd} | bs: {bs}')
                print(f'valid accuracy: {valid_acc}')
    return grid['wd_list'], acc

def logistic_regression_print_bs(fix_step = False):
    grid = {
        'lr_list':[1e-3],
        'wd_list':[1e-6],
        'bs_list':[1, 200, 2000, 20000],
    }
    acc = []
    train_data, valid_data = split_dataset()
    for lr in grid['lr_list']:
        for wd in grid['wd_list']:
            for bs in tqdm(grid['bs_list']):
                print(f'<<<<<<<<<<<<<<<<<<')
                model = LogisticRegression(n_feat=56*28, n_class=19, lr=lr, weight_decay=wd)
                model, train_loss, train_acc, valid_acc = train_model(model, train_data, batch_size=bs, max_epoch=100, valid_data=valid_data, fix_step=fix_step)
                acc.append(valid_acc)
                print(f'lr: {lr} | wd : {wd} | bs: {bs}')
                print(f'valid accuracy: {valid_acc}')
    return grid['bs_list'], acc


if __name__ == '__main__':
    lr, acc_lr = logistic_regression_print_lr(fix_step=True)
    plot_graph("LearningRate", "logistic regression", lr, acc_lr, True)

    wd, acc_wd = logistic_regression_print_wd(fix_step=True)
    plot_graph("WeightDecay", "logistic regression", wd, acc_wd, False)

    bs, acc_bs = logistic_regression_print_wd(fix_step=True)
    plot_graph("BatchSize", "logistic regression", bs, acc_bs, False)
