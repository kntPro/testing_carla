import pickle
from config import *
from train_Resnet18 import TensorImageDataset
from torch.utils.data import DataLoader, Dataset


'''
f = open(TRAFFIC_LIGHT_INT,"rb")
print(pickle.load(f))
f.close()
'''

'''
with open("data/test_label","rb") as test:
    test_label = pickle.load(test)
    print(test_label)
with open("data/train_label","rb") as train:
    train_label = pickle.load(train)
    print(train_label)
with open(TRAFFIC_LIGHT_INT,"rb") as traffic:
    all = pickle.load(traffic)
    print(all[:TRAIN_NUM])
    print(all[TRAIN_NUM:TRAIN_NUM + TEST_NUM])

a = list(range(10))
print(a[:5])
print(a[5:5+5])
'''


train_dataset = TensorImageDataset(TRAFFIC_LIGHT_INT_PATH,IMG_TRAIN_PATH)
train_dataloader = DataLoader(train_dataset)
for X,y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
