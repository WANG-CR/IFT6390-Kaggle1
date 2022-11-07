import torch
from torchvision.models.resnet import resnet18, resnet34

class Resnet:
    def __init__(self, model, n_class, lr, weight_decay):    
        if model == "resnet18":
            self.model = resnet18(num_classes=n_class)
        if model == "resnet34":
            self.model = resnet34(num_classes=n_class)
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # we choose SGD optimizer instead of Adagrad/RMSprop/Adam

    def _convert_numpy_to_torch(self, x, is_label = False):
        x = torch.from_numpy(x)
        if not is_label:
            x = x.view((-1, 1, 28, 56))
            x = x.float()
        else:
            x = torch.argmax(x, axis=-1)
        return x
    
    def train_step(self, x, y):
        self.model.train()
        loss, accuracy = self(x, y)
        loss.backward()
        self.optimizer.step()
        return loss.item(), accuracy.item()

    def __call__(self, x, y):
        x = self._convert_numpy_to_torch(x)
        labels = self._convert_numpy_to_torch(y, is_label = True)
        logits = self.model(x)
        loss = self.loss(logits, labels)
        
        pred_digit = torch.argmax(logits, axis=-1)
        accuracy = (pred_digit == labels).sum()/labels.shape[0]
        return loss, accuracy

    def predict(self, x, y=None):
        self.model.eval()
        
        x = self._convert_numpy_to_torch(x)
        logits = self.model(x)
        pred_digit = torch.argmax(logits, axis=-1)

        # in case of testing
        if y is None:
            return pred_digit
        
        # in case of validation
        labels = self._convert_numpy_to_torch(y, is_label = True)
        accuracy = ((pred_digit == labels).sum()/labels.shape[0]).item()
        return pred_digit, accuracy