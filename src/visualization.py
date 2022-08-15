import torch
import numpy as np
from src import config
from src import data_preparation
import matplotlib.pyplot as plt

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()
def visualize_model(model, dataloader, classes, num_images=4):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    plt.figure(figsize=(12, 6))
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(num_images):
                images_so_far += 1
                plt.figure(figsize=(10, 10))
                ax = plt.subplot(num_images // 2, 4, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {preds[j].detach().cpu().numpy()} \n true label :{labels[j].detach().cpu().numpy()}')
                print(labels[j].detach().cpu().numpy(),
                    data_preparation.bird_classes['class'][labels[j].detach().cpu().numpy()])

                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

