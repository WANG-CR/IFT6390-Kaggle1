import math
import numpy as np
from tqdm import tqdm

def train_model(model, train_data, batch_size, max_epoch=1000, valid_data=None, fix_step=False):
    num_samples = train_data.shape[0]
    best_valid_accuracy = 0
    loss_all = []
    train_accuracy = []
    valid_accuracy = []
    best_valid_accuracy = 0
    iter_per_epoch = math.ceil(num_samples/batch_size)
    if fix_step:
        num_epoch = math.ceil(max_epoch * batch_size / 10)
    else:
        num_epoch = max_epoch
    for i in tqdm(range(num_epoch)):
        accuracy_epoch = 0
        loss_epoch = 0
        copy_train_data = train_data
        np.random.shuffle(copy_train_data)
        # for j in tqdm(range(iter_per_epoch)):
        for j in range(iter_per_epoch):
            xtrain = copy_train_data[batch_size*j : batch_size*(j+1), :1568]
            ytrain = copy_train_data[batch_size*j : batch_size*(j+1), 1568:1587]
            loss, accuracy = model(xtrain, ytrain)
            loss_epoch += loss
            accuracy_epoch += accuracy

        loss_all.append(loss_epoch/iter_per_epoch)
        train_accuracy.append(accuracy_epoch/iter_per_epoch)
        if valid_data is not None:
            xvalid = valid_data[..., :1568]
            yvalid = valid_data[..., 1568:1587]
            _, accuracy = model.predict(xvalid, yvalid)
            valid_accuracy.append(accuracy)
            if accuracy > best_valid_accuracy:
                best_valid_accuracy = accuracy
            if accuracy < best_valid_accuracy - 0.02:
                print("it is overfitting, we apply early stop")
                break
            # print(i,"th epoch:")
            # print("Epoch Accuracy:", train_accuracy[-1])
            # print("Valid accuracy:", valid_accuracy[-1])
            # print("Loss:",loss_all[-1])
    if valid_data is not None:
        return model, loss_all, train_accuracy, best_valid_accuracy
    else:
        return model, loss_all, train_accuracy
    