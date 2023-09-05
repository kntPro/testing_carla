TRAFFIC_LIGHT_INT_PATH = "data/traffic_light_int"
LABEL_TRAIN_PATH = "data3cam/train_label"
LABEL_TEST_PATH = "data3cam/test_label"
IMG_TRAIN_PATH = "data3cam/train/"
IMG_TEST_PATH = "data3cam/test/"
IMAGE_SIZE_X = '288'
IMAGE_SIZE_Y = '288'
IMAGE_PATH = './_out_3cam'
IMAGE_NUM = 4 #モデルが一度に読み込む画像の枚数
TRAIN_NUM = int(8e1)
TEST_NUM = int(2e1)
TICK_COUNT = TRAIN_NUM + TEST_NUM 
BATCH_SIZE = 256
EPOCH = 1 
