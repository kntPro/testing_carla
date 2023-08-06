import pickle
from config import *

f = open(TRAFFIC_LIGHT_BOOLEAN,"rb")

print(pickle.load(f))
f.close()