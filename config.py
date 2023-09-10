OUT_PATH = './_out_3cam/'
IMAGE_PATH = OUT_PATH + 'images/'
LABEL_PATH = OUT_PATH + 'labels'

DATA_PATH = "data3cam/"
LABEL_TRAIN_PATH = DATA_PATH + "train_label"
LABEL_TEST_PATH = DATA_PATH + "test_label"
IMG_TRAIN_PATH = DATA_PATH + "train/"
IMG_TEST_PATH = DATA_PATH + "test/"

IMAGE_SIZE_X = '288'
IMAGE_SIZE_Y = '288'

TRAIN_NUM = int(8e2)
TEST_NUM = int(2e2)
TICK_COUNT = TRAIN_NUM + TEST_NUM 

IMAGE_NUM = 4 #モデルが一度に読み込む画像の枚数
BATCH_SIZE = 32
EPOCH = 1
