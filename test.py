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

'''
with open(LABEL_TEST_PATH,"rb") as test:
    test_label = pickle.load(test)
    print(test_label)

with open(LABEL_TRAIN_PATH,"rb") as train:
    train_label = pickle.load(train)
    print(train_label)
'''

with open(TRAFFIC_LIGHT_INT_PATH,"rb") as traffic:
    label = pickle.load(traffic)
print(type(label[TRAIN_NUM:TRAIN_NUM + TEST_NUM]))


test_dataset = TensorImageDataset(LABEL_TEST_PATH,IMG_TRAIN_PATH)
test_dataloader = DataLoader(test_dataset)
for X,y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
