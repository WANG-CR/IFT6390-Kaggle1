import numpy as np
import resnet
from resnet import Resnet
from logreg import LogisticRegression
from data_module import split_dataset
from train_model import train_model

def logistic_regression_grid_search():
    grid = {
        'lr_list':[1e-4, 1e-3, 5e-3, 1e-2],
        'wd_list':[1e-6, 1e-5, 1e-4, 0],
        'bs_list':[200, 1, 2000, 20000],
    }
    train_data, valid_data = split_dataset()
    for lr in grid['lr_list']:
        for wd in grid['wd_list']:
            for bs in grid['bs_list']:
                print(f'<<<<<<<<<<<<<<<<<<')
                model = LogisticRegression(n_feat=56*28, n_class=19, lr=lr, weight_decay=wd)
                model, train_loss, train_acc, valid_acc = train_model(model, train_data, batch_size=bs, max_epoch=100, valid_data=valid_data)
                print(f'lr: {lr} | wd : {wd} | bs: {bs}')
                print(f'valid accuracy: {valid_acc}')

def resnet_grid_search():
    grid = {
        'lr_list':[1e-4, 1e-3, 5e-3, 1e-2],
        'wd_list':[0, 1e-6, 1e-5, 1e-4],
        'bs_list':[200, 1000, 1],
    }
    train_data, valid_data = split_dataset()
    for lr in grid['lr_list']:
        for wd in grid['wd_list']:
            for bs in grid['bs_list']:
                model = Resnet('resnet18', 19, lr, wd)
                model, train_loss, train_acc, valid_acc = train_model(model, train_data, batch_size=bs, max_epoch=1500, valid_data=valid_data)
                print(f'<<<<<<<<<<<<<<<<<<')
                print(f'lr: {lr} | wd : {wd} | bs: {bs}')
                print(f'valid accuracy: {valid_acc}')

if __name__ == '__main__':
    logistic_regression_grid_search()
    resnet_grid_search()