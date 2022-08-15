import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from src import config

birds = pd.read_csv(config.BIRDS_DIR)
birds_class = pd.read_csv(config.BIRDS_CLASS_DIR)

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
                  shuffle=False,
                  num_workers=4) for x in ['train', 'test']

}

test_dataset = datasets.ImageFolder(root=config.TEST_DIR,
                                    transform=transformations_test)

test_dataloader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=True
                             )
