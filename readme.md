# Kaggle Competition 1 IFT 6390

## Author
Chuanrui Wang, matricule no. 20242243

## Package dependencies

```
numpy 
pandas
torch
matplotlib
```

Note that

- `numpy` and `pandas` are used universally for loading/processing data and representating the features.
- **Logistic regression** method is constructed from scratch and is only based on `numpy`.
- `pytorch` is only used for implementation of **ResNet** model.
- `matplotlib` are used to plot figures and graphs in the report. 


## How to run the code:
- Remark: before running the above scripts, please create a sub-repository called 'data' to put all csv files.
- `python grid_search.py` to do grid search on hyperparameters for both models

- `python predict.py` to give prediction of each model under their best hyperparameters

- `python plot.py` to generate the accuracy curves dependant on hyperparameter.

- Other `.py` files are utilized as functions / methods for above scripts. Specifically,

  - `data_module.py` provides functions that load and process dataset. This can also be executed to print the first two inputs and corresponding labels.
  - `train_model.py` defines a funtion that trains an given model with specified hyperparameters.
  - `logreg.py` defines Logistic Regression model
  - `resnet.py` defines ResNet model


  

