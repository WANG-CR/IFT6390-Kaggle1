import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

class LogisticRegression:
    def __init__(self, n_feat, n_class, lr, weight_decay):
        '''Initiate the logistic regression model
        '''
        self.weight = np.random.normal(0,0.01,(n_feat,n_class))
        self.bias = np.random.normal(0,0.001,(1,n_class))
        self.lr = lr
        self.reg = weight_decay

    def __call__(self, x, y):
        '''Execute one forward propagation step + one backward propagation step.
        '''
        N = x.shape[0]
        logits = x @ self.weight + self.bias
        pred = softmax(logits)
        self.weight = self.weight - self.lr * 1/N * (x.T@(pred - y)) - self.reg*self.weight
        self.bias = self.bias - self.lr * 1/N * (pred - y).sum(axis=0, keepdims=True) - self.reg*self.bias
        
        pred_digit = np.argmax(pred, axis=-1)
        y_digit = np.argmax(y, axis=-1)
        accuracy = (pred_digit == y_digit).sum() / N
        loss = np.sum(-y * np.log(pred + 1e-15) / pred.shape[0]) + self.reg**np.sum(np.square(self.weight))
        return loss, accuracy

    def predict(self, X, y=None):
        '''Give predictions to input feature X.
        If their corresponding label y is given, the function performs validation and output accuracy score.
        Else, the function only performs predicting and output the predicted logits. 
        '''
        hidden = X @ self.weight
        pred = softmax(hidden)
        pred_digit = np.argmax(pred, axis=-1)
        if y is None:
            return pred_digit
            
        y_digit = np.argmax(y, axis=-1)
        accuracy = (pred_digit == y_digit).mean()
        return pred_digit, accuracy