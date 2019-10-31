from .resnet import resnet50, resnet50_fc512


__model_factory = {
    'resnet50': resnet50,
    'resnet50_fc512': resnet50_fc512,
}


def build_model(name, num_classes, bias, bnneck, pretrained=True):
    """A function wrapper for building a model.
    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        bias (bool): whether to add bias.
        bnneck (bool): whether to add bnneck.
    Returns:
        nn.Module
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(name, avai_models))
    return __model_factory[name](num_classes, bias, bnneck, pretrained=pretrained)
 