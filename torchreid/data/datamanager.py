import torch

from .datasetloader import ImageDataset, QueryDataset, GalleryDataset
from .transforms import build_transforms


class ImageDataManager(object):

    def __init__(self, batch_size=32):

        transform_tr, transform_te = build_transforms()
        self.trainloader = torch.utils.data.DataLoader(
            ImageDataset('train_list.txt', transform=transform_tr),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        self.queryloader = torch.utils.data.DataLoader(
            QueryDataset(transform=transform_te),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        self.galleryloader = torch.utils.data.DataLoader(
            GalleryDataset(transform=transform_te),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

    def return_dataloaders(self):
        """Returns trainloader and testloader."""
        return self.trainloader, self.queryloader, self.galleryloader
