import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from src import config


birds = pd.read_csv('data/birds.csv')
birds_class = pd.read_csv('data/class_dict.csv')


bird_classes = pd.DataFrame({'index': birds_class['class_index'],
                            'class': birds_class['class']})

transformations_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.CenterCrop((200)),
    transforms.RandomHorizontalFlip()
])

transformations_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = {
    'train': datasets.ImageFolder(root=config.TRAIN_DIR, transform=transformations_train),
    'test': datasets.ImageFolder(root=config.VAL_DIR, transform=transformations_test)
}

dataloaders = {
    x: DataLoader(dataset[x],
                  batch_size=64,
                  shuffle=True,
                  num_workers=2) for x in ['train', 'test']

}