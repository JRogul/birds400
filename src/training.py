import torch
from src import config
import tqdm

def train_model(model, dataloaders, criterion, optimizer, num_epochs, num_model):
    early_stopping = 0
    early_loss = 3000

    for epoch in range(num_epochs):
        print('Epoch {}/ {}'.format(epoch, num_epochs - 1))

        print('-' * 15)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0

            for images, labels in tqdm(dataloaders[phase]):
                images = images.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_acc += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_acc / len(dataloaders[phase].dataset)

            if phase == 'test':
                if early_loss < epoch_loss:
                    early_stopping += 1
                    print('Early stopping {}/{}'.format(early_stopping, 3))
                    if early_stopping == 3:
                        return model

                elif early_loss > epoch_loss:
                    early_stopping = 0
                    if epoch > 30:
                        torch.save(model, 'model_trained_0{}_{}'.format(num_model, epoch))

            print('{} Loss: {:.3f} Acc: {:.3f}'.format(phase, epoch_loss, epoch_acc))
            # early_loss = epoch_loss
    return model
