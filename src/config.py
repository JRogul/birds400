import torch

TRAIN_DIR = '../data/train'
VAL_DIR = '../data/valid'
TEST_DIR = '../data/test'
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')