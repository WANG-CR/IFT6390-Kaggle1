import numpy as np
import pandas as pd
import resnet
from resnet import Resnet
from logreg import LogisticRegression
import data_module
from train_model import train_model

# given the best parameters obtained by applying grid search
# we train the model again on the full training set
# then use the trained model to predict


def predict(model, batch_size, train_data, test_data, save_as_name):
    model, _, _, = train_model(model, train_data, batch_size, max_epoch=10)
    test_label = model.predict(test_data)
    df = pd.DataFrame({'Class':test_label})
    df.index.name = 'Index'
    df.to_csv(save_as_name + ".csv")
    print('save to: ' + save_as_name + '.csv')

if __name__ == '__main__':
    train_data = data_module.preprocess_train_dataset()
    test_data = data_module.preprocess_test_dataset()
    model_logreg = LogisticRegression(n_feat=56*28, n_class=19, lr=1e-3, weight_decay=1e-6)
    predict(model_logreg, batch_size=200, train_data=train_data, test_data=test_data, save_as_name='logreg')
    model_resnet = Resnet(model = "resnet34", n_class=19, lr=1e-3, weight_decay=1e-6)
    predict(model_resnet, batch_size=200, train_data=train_data, test_data=test_data, save_as_name='resnet')