import os
import shutil
from config import *

###指定したpath（img_dir）のファイルを名前順にdata/train,data/testの２つのフォルダに分割する
###数はTRAIN＿NUMとTEST_NUM
def separate_img(img_dir):
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    abs_img_dir = os.path.abspath(img_dir)
    img_path_list = [img_dir + "/" + p for p in sorted(os.listdir(abs_img_dir))]
    for img_num in range(len(img_path_list)):
        if img_num < TRAIN_NUM:
            shutil.copy2(img_path_list[img_num],"data/train")
        elif img_num < TRAIN_NUM + TEST_NUM:
            shutil.copy2(img_path_list[img_num], "data/test")


def main():
    separate_img(IMAGE_PATH)

if __name__ == "__main__":
    main()
        

