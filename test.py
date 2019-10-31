import sys
import os
import os.path as osp
import time
import datetime
import numpy as np
import json
import torch
from torch.nn import functional as F

from default_parser import init_parser
from torchreid.data import TestImageDataManager
from torchreid.losses import CrossEntropyLoss
from torchreid.metrics import accuracy, compute_distance_matrix
from torchreid.models import build_model
from torchreid.optim import build_lr_scheduler

from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger
from torchreid.utils.tools import set_random_seed, check_isfile
from torchreid.utils.torchtools import load_pretrained_weights, save_checkpoint, open_all_layers, open_specified_layers

parser = init_parser()
args = parser.parse_args()


def main():
    global args
    set_random_seed(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    log_name = 'test.log' if args.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print('** Arguments **')
    arg_keys = list(args.__dict__.keys())
    arg_keys.sort()
    for key in arg_keys:
        print('{}: {}'.format(key, args.__dict__[key]))
    torch.backends.cudnn.benchmark = True

    datamanager = TestImageDataManager(batch_size=args.batch_size)
    trainloader, queryloader, galleryloader = datamanager.return_dataloaders()

    print('Building model: {}'.format(args.arch))
    model = build_model(args.arch, 4768, args.bias, args.bnneck, pretrained=(not args.no_pretrained))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    model.cuda()

    if args.evaluate:
        test(model, queryloader, galleryloader, args.dist_metric, args.normalize_feature)
        return

    criterion = CrossEntropyLoss(4768)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-04, betas=(0.9, 0.999))
    scheduler = build_lr_scheduler(optimizer, args.lr_scheduler, args.stepsize)

    time_start = time.time()
    print('=> Start training')
    for epoch in range(args.start_epoch, args.max_epoch):
        train(epoch, model, criterion, optimizer, trainloader)
        scheduler.step()
        if (epoch + 1) == args.max_epoch:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
            }, args.save_dir)
            test(model, queryloader, galleryloader, args.dist_metric, args.normalize_feature)
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))


def train(epoch, model, criterion, optimizer, trainloader):
    losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.train()
    if (epoch + 1) <= args.fixbase_epoch and args.open_layers is not None:
        print('* Only train {} (epoch: {}/{})'.format(args.open_layers, epoch + 1, args.fixbase_epoch))
        open_specified_layers(model, args.open_layers)
    else:
        open_all_layers(model)
    end = time.time()
    for batch_idx, (imgs, pids) in enumerate(trainloader):
        data_time.update(time.time() - end)
        imgs = imgs.cuda()
        pids = pids.cuda()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, pids)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        losses.update(loss.item(), pids.size(0))
        accs.update(accuracy(outputs, pids)[0].item())
        if (batch_idx + 1) % 20 == 0:
            num_batches = len(trainloader)
            eta_seconds = batch_time.avg * (num_batches - (batch_idx + 1) + (args.max_epoch - (epoch + 1)) * num_batches)
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                  'Lr {lr:.6f}\t'
                  'Eta {eta}'.format(
                epoch + 1, args.max_epoch, batch_idx + 1, len(trainloader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=accs,
                lr=optimizer.param_groups[0]['lr'],
                eta=eta_str))
        end = time.time()


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
