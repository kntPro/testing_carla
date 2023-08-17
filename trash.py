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