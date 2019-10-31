import torch

from .datasetloader import ImageDataset, TestQueryDataset, TestGalleryDataset
from .transforms import build_transforms


class ImageDataManager(object):

    def __init__(self, batch_size=32):

        transform_tr, transform_te = build_transforms()
        self.trainloader = torch.utils.data.DataLoader(
            ImageDataset('train_v_list.txt', transform=transform_tr, relabel=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        self.queryloader = torch.utils.data.DataLoader(
            ImageDataset('query_v_list.txt', transform=transform_te),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        self.galleryloader = torch.utils.data.DataLoader(
            ImageDataset('gallery_v_list.txt', transform=transform_te),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

    def return_dataloaders(self):
        """Returns trainloader and testloader."""
        return self.trainloader, self.queryloader, self.galleryloader


class TestImageDataManager(object):

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
            TestQueryDataset(transform=transform_te),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        self.galleryloader = torch.utils.data.DataLoader(
            TestGalleryDataset(transform=transform_te),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

    def return_dataloaders(self):
        """Returns trainloader and testloader."""
        return self.trainloader, self.queryloader, self.galleryloader
