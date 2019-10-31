import os
import glob
from torch.utils.data import Dataset

from torchreid.utils.tools import read_image


# train val
class ImageDataset(Dataset):
    def __init__(self, label, transform=None):
        self.root = '../naicdata'
        list_path = os.path.join(self.root, label)
        self.img_list = [i_id.strip() for i_id in open(list_path)]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path, id = (self.img_list[index]).split()
        img = read_image(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)
        pid = int(id)
        return img, pid


# test
class TestQueryDataset(Dataset):
    def __init__(self, transform=None):
        self.root = '../naicdata/test'
        list_path = os.path.join(self.root, 'query_a_list.txt')
        self.img_list = [i_id.strip() for i_id in open(list_path)]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path, _ = (self.img_list[index]).split()
        img = read_image(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)
        img_name = os.path.basename(img_path)
        return img, img_name


class TestGalleryDataset(Dataset):
    def __init__(self, transform=None):
        self.img_paths = glob.glob(os.path.join('../naicdata/test', 'gallery_a', '*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        img_name = os.path.basename(img_path)
        return img, img_name