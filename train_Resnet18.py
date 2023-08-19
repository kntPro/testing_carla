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
from separate import separate_label, separate_img

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)  

#separate.py
class TensorImageDataset(Dataset):
    #label_fileは"/data内のtestかtrainのパス、img_dirは画像があるフォルダのパスにする
    def __init__(self, label_file, img_dir, transform=None, target_transform=None) -> None:
        self.img_labels = self._open_label_data(label_file) 
        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)-IMAGE_NUM
    
    def __getitem__(self, idx):
        image_path_tuple = tuple(read_image(self.img_paths[i], mode=ImageReadMode.RGB).to(torch.float32) for i in range(idx,idx+IMAGE_NUM))
        image = torch.stack(image_path_tuple)
        label_set = set(torch.tensor(self.img_labels[i]) for i in range(idx,idx+IMAGE_NUM))
        label = int(1 in label_set)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def _get_img_paths(self, img_dir):
        img_dir = os.path.abspath(img_dir)
        img_paths = [img_dir+"/"+p for p in sorted(os.listdir(img_dir)) if os.path.splitext(p)[1] in [".jpg", ".jpeg", ".png", ".bmp"]]
        return img_paths
    
    def _open_label_data(self,label_file):
        abs_label_path = os.path.abspath(label_file)
        with open(abs_label_path,"rb") as label:
            l = pickle.load(label)
        return l


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
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


def main():
    train_dataset = TensorImageDataset(LABEL_TRAIN_PATH,IMG_TRAIN_PATH)
    test_dataset = TensorImageDataset(LABEL_TEST_PATH,IMG_TEST_PATH)
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    model = get_resnet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for t in range(EPOCH):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
    


if __name__ == '__main__':
    main()