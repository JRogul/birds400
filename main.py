import torch.nn as nn
import torch
from src import data_preparation
from src import training
from src import config
import warnings
warnings.filterwarnings('ignore')
criterion = nn.CrossEntropyLoss()
NUM_EPOCHS = 50

def choose_opt(num_model):
    if num_model == 1:
        optimizer = torch.optim.SGD(model.classifier.parameters(),
                                lr = 0.001,
                                momentum=0.9)

    elif num_model == 2:
        optimizer = torch.optim.SGD(model.fc.parameters(),
                                lr = 0.001,
                                momentum=0.9)
    elif num_model == 3:
        optimizer = torch.optim.SGD(model.classifier[6].parameters(),
                                lr = 0.001,
                                momentum=0.9)

    return optimizer

if __name__ ==  '__main__':
    for num_model in range(1, 4):
        print(config.DEVICE)
        model = torch.load('models/model_{}'.format(num_model))

        model = model.to(config.DEVICE)

        optimizer = choose_opt(num_model)

        training.train_model(model,
                             data_preparation.dataloaders,
                             criterion,
                             optimizer,
                             NUM_EPOCHS,
                             num_model)