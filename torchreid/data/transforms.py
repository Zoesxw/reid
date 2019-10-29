from torchvision.transforms import Normalize, Compose, RandomHorizontalFlip, ToTensor


def build_transforms():
    """Builds train and test transform functions.
    """

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_tr = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
    ])

    transform_te = Compose([
        ToTensor(),
        normalize,
    ])

    return transform_tr, transform_te
