import torch 
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms 
from torchvision.io import read_image, write_png
from torchvision.io import ImageReadMode
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import os
from config import *
import pickle
from separate import separate_label, separate_img
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime

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
        image = torch.concat(image_path_tuple)
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

#front,left,rightの3方向のカメラ画像をDatasetにするクラス
class ThreeImageToTensorDataset(Dataset):
    #label_fileは"/data内のtestかtrainのパス、img_dirは画像があるフォルダのパスにする
    def __init__(self, label_file, img_dir, transform=ResNet18_Weights.IMAGENET1K_V1.transforms, target_transform=None) -> None:
        self.img_labels = self._open_label_data(label_file) 
        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels[:,0])-IMAGE_NUM
    
    def __getitem__(self, idx):
        #image_path_tuple = tuple(read_image(self.img_paths[i], mode=ImageReadMode.RGB).to(torch.float32) for i in range(idx,idx+IMAGE_NUM))
        img_list = []
        for i in range(idx, idx+IMAGE_NUM):
            for img in self.img_paths[i]:
                img_list.append(read_image(img, mode=ImageReadMode.RGB).to(torch.float32))
        images = torch.concat(img_list)
        label_list = torch.tensor(np.array([self.img_labels[i] for i in range(idx,idx+IMAGE_NUM)]))
        label = torch.max(label_list,dim=0).values.to(torch.float32) #4フレーム中に一つでも１があったら（１フレームでも信号機が赤になっていたら）１になる
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return images, label
    
    def _get_img_paths(self, img_dir):
        img_dir = os.path.abspath(img_dir)
        img_paths = [img_dir+"/"+p for p in sorted(os.listdir(img_dir)) if os.path.splitext(p)[1] in [".jpg", ".jpeg", ".png", ".bmp"]]

        front_path = [f for f in img_paths if "front" in f]
        left_path = [f for f in img_paths if "left" in f]
        right_path = [f for f in img_paths if "right" in f]
        
        cam_path_list = []
        for i in range(min(len(front_path),min(len(right_path),len(left_path)))):
            cam_path_list.append([left_path[i],front_path[i],right_path[i]])
       
        return cam_path_list
    
    def _open_label_data(self,label_file):
        abs_label_path = os.path.abspath(label_file)
        with open(abs_label_path,"rb") as label:
            l = pickle.load(label)
        return l


def get_resnet(num_label: int=2) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    class outLayer(nn.Module):
        def __init__(self, in_units):
            super(outLayer,self).__init__()
            self.l1 = nn.Linear(in_units,32)
            self.out1 = nn.Linear(32,1)
            self.out2 = nn.Linear(32,1)

        def forward(self,x):
            h = nn.functional.relu(self.l1(x))
            out1 = nn.functional.relu(self.out1(h))
            out2 = nn.functional.relu(self.out2(h))

            return out1,out2

   # ここで更新する部分の重みは初期化される
    model.conv1 = nn.Conv2d(
        in_channels=3*3*IMAGE_NUM,
        out_channels=64,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False
   )

    #out_layer = outLayer(in_units=1000)  #model.fcの出力数
    model.fc = nn.Sequential(
        nn.Linear(in_features=512,out_features=num_label),
        nn.Sigmoid()
    )
    
    return model

def train(dataloader, model, loss_fn, optimizer, epoch ,on_write:bool=False):
    size = len(dataloader.dataset)
    model.train()
    bool_write = (on_write and (epoch+1)%5==0)
    if bool_write:
        day = datetime.now().strftime(f"%Y-%m-%d/%H:%M:%S_epoch:{epoch+1}")
        writer = SummaryWriter(log_dir="runs/"+day)

    for batch, (X, y) in enumerate(tqdm(dataloader)):
        # Compute prediction error
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y)

        if(bool_write):
            writer.add_scalar("loss",loss,batch)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        
    loss, current = loss.item(), (batch + 1) * len(X)
    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    if(bool_write):
        writer.close()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, tl_correct, is_correct = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            tl_correct += (torch.round(pred[:,0]).type(torch.int64) == y[:,0]).type(torch.float).sum().item()
            is_correct += (torch.round(pred[:,1]).type(torch.int64) == y[:,1]).type(torch.float).sum().item()
            test_loss /= num_batches
    tl_correct /= size
    is_correct /= size
    time.sleep(2.)
    print(f"Test Error: \n TrafficLight Accuracy: {(100*tl_correct):>0.1f}%,  Intersection Accuracy: {(100*is_correct):>0.1f}% \nAvg loss: {test_loss:>8f} \n")


def main():
    train_dataset = ThreeImageToTensorDataset(LABEL_TRAIN_PATH,IMG_TRAIN_PATH)
    test_dataset = ThreeImageToTensorDataset(LABEL_TEST_PATH,IMG_TEST_PATH)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = get_resnet().to(device)
    loss_fn = nn.functional.binary_cross_entropy
    optimizer = torch.optim.Adam(model.parameters())
    
    for t in tqdm(range(EPOCH)):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, epoch=t, on_write=True)
        test(test_dataloader, model, loss_fn)

    torch.save(model.state_dict(),"model/trained_resnet18_3cam")
    print("Done!")
    


if __name__ == '__main__':
    main()