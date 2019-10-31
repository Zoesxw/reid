import argparse


def init_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--gpu-devices', type=str, default='1',
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--save-dir', type=str, default='log',
                        help='path to save log and model weights')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument('--arch', type=str, default='resnet50',
                        help='model architecture')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='manual epoch number (useful when restart)')
    parser.add_argument('--max-epoch', type=int, default=60,
                        help='maximum epochs to run')
                        
    parser.add_argument('--fixbase-epoch', type=int, default=5,
                        help='number of epochs to fix base layers')
    parser.add_argument('--open-layers', type=str, nargs='+', default=['classifier'],
                        help='open specified layers for training while keeping others frozen')

    parser.add_argument('--lr-scheduler', type=str, default='multi_step',
                        help='learning rate scheduler (see lr_schedulers.py)')
    parser.add_argument('--stepsize', type=int, default=[20, 40], nargs='+',
                        help='stepsize to decay learning rate')

    parser.add_argument('--dist-metric', type=str, default='cosine',
                        help='distance metric')
    parser.add_argument('--normalize-feature', action='store_true',
                        help='normalize feature vectors before calculating distance')
    parser.add_argument('--bias', action='store_true',
                        help='classifier bias')
    parser.add_argument('--bnneck', action='store_true',
                        help='bnneck')
    # ************************************************************
    # Test settings
    # ************************************************************
    parser.add_argument('--no-pretrained', action='store_true',
                        help='do not load pretrained weights')
    parser.add_argument('--load-weights', type=str, default='',
                        help='load pretrained weights but ignore layers that do not match in size')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate only')

    return parser
