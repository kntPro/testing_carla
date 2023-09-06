import pickle
from config import *
from train_Resnet18 import TensorImageDataset,ThreeImageToTensorDataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from train_Resnet18 import get_resnet
import torch
from torch import nn
from torchvision.io import read_image,ImageReadMode
import os
import re
from torchvision.models import resnet18



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
'''
test_dataset = TensorImageDataset(LABEL_TEST_PATH,IMG_TRAIN_PATH)
test_dataloader = DataLoader(test_dataset)
for X,y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
'''

'''
with open(TRAFFIC_LIGHT_INT_PATH,"rb") as traffic:
    label = pickle.load(traffic)

print(1 in label)
print(0 in label)
'''

'''
model = get_resnet()
with open("model_architecture.txt","w") as f:
    print(model,file=f)
'''

'''
a = torch.ones(2,3)
d = torch.concat((a,a,a,a),)
#b = tuple(d[i] for i in range(4))
#c = torch.stack(b)

print(a.shape)
print(d.shape)
#print(c.shape)
'''

'''
with open(TRAFFIC_LIGHT_INT_PATH,"rb") as t:
    traffic = pickle.load(t)
 
for i in range(len(traffic)-1):
    if not traffic[i] == traffic[i+1]:
        print(i)

print(len(traffic))
'''

'''
a = tuple(read_image(path=os.path.join(IMAGE_PATH,os.listdir(IMAGE_PATH)[i]), mode=ImageReadMode.RGB) for i in range(4))
b = torch.concat(a)
print(b.shape)
'''

'''
with open(LABEL_TEST_PATH,"rb") as t:
    test = pickle.load(t)
with open(LABEL_TRAIN_PATH,"rb") as t:
    train = pickle.load(t)

count =0
for i in range(len(train)-1):
    if not train[i] == train[i+1]:
        print(i)
        count+=1

print(count)
'''

'''
loss_fn = nn.MSELoss()
writer = SummaryWriter()
loss = []
for i in range(int(1e4)):
    loss = loss_fn(torch.randn(1),torch.tensor(1))
    writer.add_scalar("random_closed",loss,i)
writer.close()
'''

'''
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

        self.l1 = nn.Linear(100, 10)
        self.l2 = nn.Linear(10, 10)

    def forward(self, x):
        h1 = torch.nn.functional.tanh(self.l1(x))
        return self.l2(h1)
m = DummyModel()
m = m.cuda()
torch.save(m,"model/test_model")
'''

#m = torch.load("model/test_model")
# print(m)


'''
with open("test.txt","wt") as f:
     paths = get_img_paths(IMAGE_PATH)
     paths = map(lambda x:x+"\n",paths)
     f.writelines(paths)
'''
'''
a , alist = get_img_paths(IMAGE_PATH)
for l in alist:
    a = re.search(r'\d+',l[0]).group()
    b = re.search(r'\d+',l[1]).group()
    c = re.search(r'\d+',l[2]).group()
    if a == b and b == c:
        pass
    else:
         print("%s, %s, %s"%(l[0],l[1],l[2]))
'''
'''
dataset = ThreeImageToTensorDataset(TRAFFIC_LIGHT_INT_PATH,IMAGE_PATH)
dataloader = DataLoader(dataset, batch_size=1)
with open("test.txt","w") as f:
    for batch,(X,y) in enumerate(dataloader):
        print(f"bacth:{batch}, X:{X.size()}, y:{y}" ,file=f)

with open(TRAFFIC_LIGHT_INT_PATH,"rb") as f:
    print(len(pickle.load(f)))
'''

def get_resnet(num_classes: int=2) -> nn.Module:
   # ImageNetで事前学習済みの重みをロード
    model = resnet18(weights='DEFAULT')

   # ここで更新する部分の重みは初期化される
    model.conv1 = nn.Conv2d(
        in_channels=3*IMAGE_NUM,
        out_channels=64,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False
   )

    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    print(type(model))
    return model

print(get_resnet())