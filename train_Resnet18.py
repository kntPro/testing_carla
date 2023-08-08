import torch 
from torch import nn
from torchvision.models import resnet18
import torchvision.transforms as transforms 
from torchvision.io import read_image, write_png
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode
import os
from config import *
import pickle
from separateImg import separate_img

'''
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)


test_dataloader = DataLoader(test_data, batch_size=64)
'''

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class imageDataset(Dataset):
    # パスとtransformの取得
  def __init__(self, img_dir, transform=None):
        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform

  # データの取得
  def __getitem__(self, index):
      path = self.img_paths[index]
      img = read_image(path,mode=ImageReadMode.RGB)
      #if self.transform is not None:
          #img = self.transform(img)
      return img
  
  # パスの取得
  def _get_img_paths(self, img_dir):
      img_dir = os.path.abspath(img_dir)
      img_paths = [img_dir+"/"+p for p in sorted(os.listdir(img_dir)) if os.path.splitext(p)[1] in [".jpg", ".jpeg", ".png", ".bmp"]]
      return img_paths

  # ながさの取得
  def __len__(self):
      return len(self.img_paths)



def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.cuda()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def get_resnet(num_classes: int=10) -> nn.Module:
   # ImageNetで事前学習済みの重みをロード
    model = resnet18(weights='DEFAULT')
    model.to(device)
   # ここで更新する部分の重みは初期化される
    model.conv1 = nn.Conv2d(
        in_channels=1,
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


def main():
    #model = get_resnet()
    #loss_fn = nn.CrossEntropyLoss()
    #epochs = 5
    #transform = transforms.Compose([transforms.ToTensor()])
    #train_data, test_data= imageDataset(IMAGE_PATH)
    #separate_img(IMAGE_PATH)
    train_dataset = imageDataset("data/train")
    test_dataset = imageDataset("data/test")
    train_data = DataLoader(train_dataset,batch_size=BATCH_SIZE)
    test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    for i in train_data:
        print(i.shape)
    '''
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    with open(TRAFFIC_LIGHT_BOOLEAN,"rb") as f:
        supervise_data = pickle.load(f)

    #test(test_dataloader, model, loss_fn)
    print(train_dataloader.shape)
    print(test_dataloader.shape)
    print(supervise_data)
    print("Done!")
'''

    


if __name__ == '__main__':
    main()