#改良して使わなくなったコード置き場
class imageDataset(Dataset):
    # パスとtransformの取得
  def __init__(self, img_dir, transform=None):
        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform

  # データの取得
  def __getitem__(self, index):
      path = self.img_paths[index]
      img = read_image(path,mode=ImageReadMode.RGB)
      #if self.transform is not None:
          #img = self.transform(img)
      return img
  
  # パスの取得
  def _get_img_paths(self, img_dir):
      img_dir = os.path.abspath(img_dir)
      img_paths = [img_dir+"/"+p for p in sorted(os.listdir(img_dir)) if os.path.splitext(p)[1] in [".jpg", ".jpeg", ".png", ".bmp"]]
      return img_paths

  # ながさの取得
  def __len__(self):
      return len(self.img_paths)
      


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
