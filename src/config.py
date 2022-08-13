import torch

TRAIN_DIR = 'data/train'
VAL_DIR = 'data/valid'
TEST_DIR = 'data/test'
BIRDS_DIR = 'data/birds.csv'
BIRDS_CLASS_DIR = 'data/class_dict.csv'
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')