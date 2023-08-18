import os
import shutil
from config import *
import pickle

###指定したpath（img_dir）のファイルを名前順にdata/train,data/testの２つのフォルダに分割する
###数はTRAIN＿NUMとTEST_NUM
def separate_img(img_dir):
    os.makedirs(IMG_TRAIN_PATH, exist_ok=True)
    os.makedirs(IMG_TEST_PATH, exist_ok=True)
    abs_img_dir = os.path.abspath(img_dir)
    img_path_list = [img_dir + "/" + p for p in sorted(os.listdir(abs_img_dir))]
    for img_num in range(len(img_path_list)):
        if img_num < TRAIN_NUM:
            shutil.copy2(img_path_list[img_num],IMG_TRAIN_PATH)
        elif img_num < TRAIN_NUM + TEST_NUM:
            shutil.copy2(img_path_list[img_num], IMG_TEST_PATH)

def separate_label(annotation_file):
    with open(annotation_file,"rb") as f:
        labels = pickle.load(f)
    train_label_list = labels[:TRAIN_NUM]
    test_label_list = labels[TRAIN_NUM:TRAIN_NUM + TEST_NUM]
    with open(LABEL_TRAIN_PATH,"wb") as train_label_file:
        pickle.dump(train_label_list,train_label_file)
    with open(LABEL_TEST_PATH,"wb") as test_label_file:
        pickle.dump(test_label_list,test_label_file)


def main():
    separate_img(IMAGE_PATH)
    separate_label(TRAFFIC_LIGHT_INT_PATH)

if __name__ == "__main__":
    main()
        

