import torch
#paths for data
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/valid'
TEST_DIR = 'data/test'
BIRDS_DIR = 'data/birds.csv'
BIRDS_CLASS_DIR = 'data/class_dict.csv'
#used for training on gpu
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')