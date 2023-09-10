import os
import shutil
from config import *
import pickle
import numpy as np

###指定したpath（img_dir）のファイルを名前順にdata/train,data/testの２つのフォルダに分割する
###数はTRAIN＿NUMとTEST_NUM
def separate_img(img_dir):
    os.makedirs(IMG_TRAIN_PATH, exist_ok=True)
    os.makedirs(IMG_TEST_PATH, exist_ok=True)
    abs_img_dir = os.path.abspath(img_dir)
    img_path_list = [img_dir + "/" + p for p in sorted(os.listdir(abs_img_dir))]
    front_path_list = [p for p in img_path_list if "front" in p]
    left_path_list = [p for p in img_path_list if "left" in p]
    right_path_list = [p for p in img_path_list if "right" in p]
    for img_num in range(min(len(front_path_list),min(len(left_path_list),len(right_path_list)))):
        if img_num < TRAIN_NUM:
            shutil.copy2(front_path_list[img_num],IMG_TRAIN_PATH)
            shutil.copy2(left_path_list[img_num],IMG_TRAIN_PATH)
            shutil.copy2(right_path_list[img_num],IMG_TRAIN_PATH)
        elif img_num < TRAIN_NUM + TEST_NUM:
            shutil.copy2(front_path_list[img_num],IMG_TEST_PATH)
            shutil.copy2(left_path_list[img_num],IMG_TEST_PATH)
            shutil.copy2(right_path_list[img_num],IMG_TEST_PATH)

def separate_label(annotation_file,train_path,test_path):
    with open(annotation_file,"rb") as f:
        labels = dict(pickle.load(f))

    #辞書に入っている配列を順番にリストにする
    #出力は(n,1,len(key))の次元を持つ配列

    value_list = [labels[k] for k in labels.keys()]
    label_list = list(map(lambda *x:list(x), *value_list))
    with open(train_path,"wb") as train:
        pickle.dump(np.array(label_list[:TRAIN_NUM]), train)
    with open(test_path,"wb") as test:
        pickle.dump(np.array(label_list[TRAIN_NUM:TRAIN_NUM+TEST_NUM]), test)



def main():
    os.makedirs(DATA_PATH, exist_ok=True)
    separate_img(IMAGE_PATH)
    separate_label(LABEL_PATH, LABEL_TRAIN_PATH,LABEL_TEST_PATH)

if __name__ == "__main__":
    main()
        

