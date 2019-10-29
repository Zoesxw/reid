import sys
import os
import os.path as osp
import time
import datetime
import torch

from default_parser import init_parser
from torchreid.data import ImageDataManager
from torchreid.losses import CrossEntropyLoss
from torchreid.metrics import accuracy
from torchreid.models import build_model
from torchreid.optim import build_lr_scheduler

from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger
from torchreid.utils.tools import set_random_seed, check_isfile
from torchreid.utils.torchtools import load_pretrained_weights, save_checkpoint

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

    datamanager = ImageDataManager(batch_size=args.batch_size)
    trainloader, queryloader, galleryloader = datamanager.return_dataloaders()

    print('Building model: {}'.format(args.arch))
    model = build_model(args.arch, 4768, args.bias, args.bnneck)

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    model.cuda()

    criterion = CrossEntropyLoss(4768)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-04, betas=(0.9, 0.999))
    scheduler = build_lr_scheduler(optimizer, args.lr_scheduler, args.stepsize)

    time_start = time.time()
    print('=> Start training')
    for epoch in range(args.start_epoch, args.max_epoch):
        train(epoch, model, criterion, optimizer, trainloader)
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
            }, args.save_dir)
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))


def train(epoch, model, criterion, optimizer, trainloader):
    losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.train()
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


if __name__ == '__main__':
    main()
