
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
    import torch.nn as nn
    import torch
    from src import data_preparation
    from src import training
    from src import config
    from src import visualization
    import warnings

    model = torch.load('models/model01')
    model = model.to(config.DEVICE)
    visualization.visualize_model(model, data_preparation.test_dataloader, data_preparation.bird_classes)