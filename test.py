import pickle
from config import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18
import torch
from torch import nn
from torch.functional import F
from torchvision.io import read_image,ImageReadMode
import os
import re
import carla
import random
import itertools
from queue import Queue
import numpy as np
import timeit
from train_Resnet18 import ThreeImageToTensorDataset, get_resnet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)  
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

#収集した画像のチェック
'''
def get_img_paths(img_dir):
    img_dir = os.path.abspath(img_dir)
    img_paths = [p for p in sorted(os.listdir(img_dir)) if os.path.splitext(p)[1] in [".jpg", ".jpeg", ".png", ".bmp"]]

    front_path = [f for f in img_paths if "front" in f]
    left_path = [f for f in img_paths if "left" in f]
    right_path = [f for f in img_paths if "right" in f]
        
    cam_path_list = []
    for i in range(min(len(front_path),min(len(right_path),len(left_path)))):
        cam_path_list.append([left_path[i],front_path[i],right_path[i]])
    print("%s, %s, %s"%(len(left_path),len(front_path),len(right_path)))


    for l in cam_path_list:
        a = re.search(r'\d+',l[0]).group()
        b = re.search(r'\d+',l[1]).group()
        c = re.search(r'\d+',l[2]).group()
        if a == b and b == c:
            pass
        else:
            print("%s, %s, %s"%(l[0],l[1],l[2]))
    print("end")
get_img_paths(IMAGE_PATH)
'''
'''
def run_carla():
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    world_map = world.get_map()
    tm = client.get_trafficmanager(8000)

    os.makedirs("./misc/junction/",exist_ok=True )
    #f = open("waypoint_in_junction","wb")

    try:
        original_settings = world.get_settings()
        settings = world.get_settings()

        settings.fixed_delta_seconds = 0.017 
        settings.synchronous_mode = True
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)

        blueprint_library = world.get_blueprint_library()

        vehicle_bp = blueprint_library.find("vehicle.audi.tt")
        if vehicle_bp.has_attribute('color'):
            color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
            vehicle_bp.set_attribute('color',color)
        vehicle_transform = random.choice(world_map.get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
        #tm.vehicle_percentage_speed_difference(vehicle, 100.)

        sensor_queue = Queue()
        sensor_list = []
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=2.0,z=2.0))
        camera = world.spawn_actor(camera_bp,camera_transform, attach_to=vehicle)
        camera.listen(lambda data:save_image(data))
        def save_image(data):
            data.save_to_disk('./misc/junction/%6d.png'%(data.frame))
            sensor_queue.put(data.frame)
        sensor_list.append(camera)

        for i in itertools.count(0):
            world.tick()
            vehicle.set_autopilot(True)

            try:
                sensor_frame = sensor_queue.get(True, 1.0)
                print("     Frame:%d"%(sensor_frame))
            except Empty:
                print("No picture exist")

            if world_map.get_waypoint(vehicle.get_transform().location).is_junction:
                print("detect junction")
                break
            

    finally:
        world.apply_settings(original_settings)
        vehicle.destroy()
        print("destroyed vehicle")
        for s in sensor_list:
            s.destoy()
        #f.close()
        print("closed file")

try:
    run_carla()
except KeyboardInterrupt:
    print("Correctly End")
'''
'''
img_dir = os.path.abspath(IMAGE_PATH)
img_paths = [p for p in sorted(os.listdir(img_dir)) if os.path.splitext(p)[1] in [".jpg", ".jpeg", ".png", ".bmp"]]

front_path = [f for f in img_paths if "front" in f]
left_path = [f for f in img_paths if "left" in f]
right_path = [f for f in img_paths if "right" in f]
print(f"left:{len(left_path)}, front:{len(front_path)}, right:{len(right_path)}")
'''

def check_dataset():
    os.makedirs("misc",exist_ok=True)
    dataset = ThreeImageToTensorDataset(LABEL_TEST_PATH,IMG_TEST_PATH)
    dataloader = DataLoader(dataset, batch_size=1)
    print(dataloader.__len__())
    with open("misc/dataset.txt","wt") as f:
        for batch,(X,y) in enumerate(dataloader):
            print(f"bacth:{batch}, X:{X.size()}, y:{y}" ,file=f)

    with open(LABEL_PATH,"rb") as f:
        print(len(pickle.load(f)))



def check_labels(label_path:str,log_file:str):
    np.set_printoptions(threshold=np.inf)
    with open(label_path,"rb") as l:
        label = pickle.load(l)
        with open(log_file+".txt","wt") as f:
            print(type(label), file=f)
            print(label.shape, file=f)
            print(label, file=f)
        
#check_labels(LABEL_TEST_PATH,log_file="misc/testLabel")
#check_labels(LABEL_TRAIN_PATH, log_file="misc/trainLabel")

def is_equal_labelDict_to_labelList():
    label_dict = pickle.load(open(LABEL_PATH,"rb"))
    tr = pickle.load(open(LABEL_TRAIN_PATH,"rb"))
    te = pickle.load(open(LABEL_TEST_PATH,"rb"))

    label_list = np.append(tr,te, axis=0)
    print(type(te),te.shape)
    print(type(tr),tr.shape)
    print(type(label_dict["intsersection"]),label_dict["intsersection"].shape)
    print(type(label_list),label_list.shape)
   
    print(np.array_equal(label_dict["traffic_light"], list(label_list[:,0])))
    print(np.array_equal(label_dict["intsersection"], list(label_list[:,1])))

def check_label_data():
    np.set_printoptions(threshold=np.inf)
    abs_train_path = os.path.abspath(LABEL_TRAIN_PATH)
    abs_test_path = os.path.abspath(LABEL_TEST_PATH)
    train_label=open(abs_train_path,"rb") 
    test_label=open(abs_test_path,"rb")    
    train = pickle.load(train_label)
    test = pickle.load(test_label)
    with open("misc/label.txt","wt") as f:
        print(f"train.shape{train.shape}",file=f)
        print(f"test.shape{test.shape}",file=f)
    train_label.close()
    test_label.close()    

#carlaから拾集した画像の枚数を調べる
def check_image_num():
    image_list = os.listdir(IMAGE_PATH)
    front_list = [f for f in image_list if 'front' in f]
    left_list = [f for f in image_list if 'left' in f]
    right_list = [f for f in image_list if 'right' in f]
    
    print("all:",len(image_list))
    print('front:',len(front_list))
    print('left:',len(left_list))
    print('right:',len(right_list))

def print_model_architecture(model:nn.Module):
    with open('misc/model_architecture.txt','wt') as f:
        print(model,file=f)

def check_out_resnet18(model:nn.Module):
    input = torch.randint(low=0,high=1,size=(4,36,288,288),dtype=torch.float32)
    out = model(input)
    print(out)
    for o in out:
        print(o.size())


def check_train(model, loss_fn, optimizer, on_write:bool=False):
    model.train()
    x = torch.randint(low=0,high=1,size=(4,3,288,288),dtype=torch.float32)
    y = torch.randint(low=0,high=1,size=(4,2), dtype=torch.uint8)

    # Compute prediction error
    pred1, pred2 = model(x)
    loss1 = loss_fn(pred1, y[:,0])
    loss2 = loss_fn(pred2, y[:,1])
    loss = loss1+loss2

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss= loss.item()
    print(f"loss: {loss:>7f}")

def try_train():
    model = get_resnet()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters())
    check_train(model,loss_fn=loss,optimizer=optimizer)
    
def test_model():
    model = nn.Sequential(
        nn.Linear(2,32),
        nn.ReLU(),
        nn.Linear(32,2),
        nn.Sigmoid()
    )

    input = torch.randint(low=0,high=100,size=(4,2)).to(dtype=torch.float32)
    out = model(input)
    print(input)
    print(model)
    print(out)

def test_summaary_writer():
    writer = SummaryWriter(log_dir="misc/"+datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    writer.add_graph(model=get_resnet(),input_to_model=torch.rand(size=(1,36,int(IMAGE_SIZE_X),int(IMAGE_SIZE_Y))))
    writer.add_scalar("test",np.array([1,2,3,4]),np.array([1,2,3,4]))
    writer.close()

test_summaary_writer()