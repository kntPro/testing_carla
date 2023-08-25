import sys
import cv2
import os
from config import *

# encoder(for mp4)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# output file name, encoder, fps, size(fit to image size)
video = cv2.VideoWriter('trainDataVideo.mp4', fourcc, 60.0, (int(IMAGE_SIZE_X),int(IMAGE_SIZE_Y)))

if not video.isOpened():
    print("can't be opened")
    sys.exit()

for i,path in enumerate([IMG_TRAIN_PATH+"/"+p for p in sorted(os.listdir(IMG_TRAIN_PATH)) if os.path.splitext(p)[1] in [".jpg", ".jpeg", ".png", ".bmp"]]):
    # hoge0000.png, hoge0001.png,..., hoge0090.png
    img = cv2.imread(path)

    # can't read image, escape
    if img is None:
        print("can't read")
        break

    # add
    video.write(img)
    print(i)

video.release()
print('written')
