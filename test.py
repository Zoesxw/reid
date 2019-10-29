import os
import time
import numpy as np
import json
import torch
from torch.nn import functional as F

from default_parser import init_parser
from torchreid.data import ImageDataManager
from torchreid.metrics import compute_distance_matrix
from torchreid.models import build_model

from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.tools import check_isfile
from torchreid.utils.torchtools import load_pretrained_weights

parser = init_parser()
args = parser.parse_args()


def main():
    global args
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    torch.backends.cudnn.benchmark = True

    datamanager = ImageDataManager(batch_size=args.batch_size)
    trainloader, queryloader, galleryloader = datamanager.return_dataloaders()

    print('Building model: {}'.format(args.arch))
    model = build_model(args.arch, 4768, args.bias, args.bnneck)

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    model.cuda()
    test(model, queryloader, galleryloader, args.dist_metric, args.normalize_feature)


def test(model, queryloader, galleryloader, dist_metric, normalize_feature):
    batch_time = AverageMeter()
    model.eval()
    with torch.no_grad():
        print('Extracting features from query set ...')
        qf, q_names = [], []
        for batch_idx, (imgs, img_names) in enumerate(queryloader):
            imgs = imgs.cuda()
            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            qf.append(features)
            q_names.extend(img_names)
        qf = torch.cat(qf, 0)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_names = [], []
        for batch_idx, (imgs, img_names) in enumerate(galleryloader):
            imgs = imgs.cuda()
            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            gf.append(features)
            g_names.extend(img_names)
        gf = torch.cat(gf, 0)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

    if normalize_feature:
        print('Normalzing features with L2 norm ...')
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)

    print('Computing distance matrix with metric={} ...'.format(dist_metric))
    distmat = compute_distance_matrix(qf, gf, dist_metric)
    distmat = distmat.numpy()
    indices = np.argsort(distmat, axis=1)

    rank = {}
    for q_idx in range(qf.size(0)):
        q_name = q_names[q_idx]
        im_list = []
        for i in range(200):
            g_idx = indices[q_idx, i]
            g_name = g_names[g_idx]
            im_list.append(g_name)
        rank[q_name] = im_list

    with open("result.json", "w") as f:
        json.dump(rank, f)
        print('done')


if __name__ == '__main__':
    main()
