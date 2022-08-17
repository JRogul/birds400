if __name__ ==  '__main__':
    import torch.nn as nn
    import torch
    from torchvision import models
    import pandas as pd
    import numpy as np
    from scipy import stats
    from src import data_preparation
    from src import training
    from src import config
    from src import visualization
    import warnings



    model1 = torch.load('models/densenet')
    model2 = torch.load('models/resnet34')
    model3 = torch.load('models/vgg16')
    criterion = nn.CrossEntropyLoss()

    predictions = np.array(training.test_loop(model1, data_preparation.test_dataloader, criterion))
    predictions2 = np.array(training.test_loop(model2, data_preparation.test_dataloader, criterion))
    predictions3 = np.array(training.test_loop(model3, data_preparation.test_dataloader, criterion))

    predictions = predictions.reshape(predictions.shape[0], 1)
    predictions2 = predictions2.reshape(predictions2.shape[0], 1)
    predictions3 = predictions3.reshape(predictions3.shape[0], 1)


    all_predictions = np.concatenate((predictions,
                                      predictions2,
                                      predictions3), axis=1)

    final_predictions = stats.mode(all_predictions, axis=1)[0]

    labels = []
    for image, label in (data_preparation.test_dataloader):
        labels.extend(label.cpu().detach().numpy())

    predictions = predictions.squeeze(axis=1)
    predictions2 = predictions2.squeeze(axis=1)
    predictions3 = predictions3.squeeze(axis=1)
    final_predictions = final_predictions.squeeze(axis=1)

    results1 = round(np.sum(
        predictions == labels) / (data_preparation.test_dataloader.__len__() * 32), 3)
    results2 = round(np.sum(predictions2 == labels) / (data_preparation.test_dataloader.__len__() * 32), 3)
    results3 = round(np.sum(predictions3 == labels) / (data_preparation.test_dataloader.__len__() * 32), 3)
    all_results = round(np.sum(final_predictions == labels) / (data_preparation.test_dataloader.__len__() * 32), 3)

    final_results = pd.DataFrame({'model': ['densenet', 'resnet34', 'vgg16', 'ensambled_models'],
                                  'accuracy': [results1, results2, results3, all_results]})

    final_results.to_csv('final_results.csv', index=False)