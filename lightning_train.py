from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from train_Resnet18 import ThreeImageToTensorDataset, get_resnet
from config import *

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)  

class LitModel(L.LightningModule):
    def __init__(self,model):
        super().__init__()
        self.model = model.to(device)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        pred = self.model(x)
        loss = F.binary_cross_entropy(pred, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = F.binary_cross_entropy(pred, y)
        tl_correct = 0.0
        is_correct = 0.0
        tl_correct += (torch.round(pred[:,0]).type(torch.int64) == y[:,0]).type(torch.float).sum().item()
        is_correct += (torch.round(pred[:,1]).type(torch.int64) == y[:,1]).type(torch.float).sum().item()

        self.log("test_loss", loss)
        self.log(f"TrafficLight Accuracy", (100*tl_correct))
        self.log(f"Intersection Accuracy", (100*is_correct))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class MyDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.train_dataset = ThreeImageToTensorDataset(LABEL_TRAIN_PATH,IMG_TRAIN_PATH)
        self.test_dataset = ThreeImageToTensorDataset(LABEL_TEST_PATH, IMG_TEST_PATH)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=12)



model = LitModel(get_resnet())
trainer = L.Trainer(max_epochs=5)
train_dataloder = DataLoader(ThreeImageToTensorDataset(LABEL_TRAIN_PATH,IMG_TRAIN_PATH))
test_dataloder = DataLoader(ThreeImageToTensorDataset(LABEL_TEST_PATH,IMG_TEST_PATH))
dataloader = MyDataModule()
trainer.fit(model,train_dataloaders=train_dataloder)
trainer.test(model,dataloaders=test_dataloder)