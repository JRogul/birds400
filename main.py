if __name__ ==  '__main__':
    import torch.nn as nn
    import torch
    from torchvision import models
    import numpy as np
    from src import data_preparation
    from src import training
    from src import config
    from src import visualization
    import warnings


    model = torch.load('models/densenet')
    model = model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    predictions = np.array(training.test_loop(model, data_preparation.test_dataloader, criterion))
    #visualization.visualize_model(model, data_preparation.test_dataloader, data_preparation.bird_classes)

