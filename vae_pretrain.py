from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
import wandb
import os
import time
import argparse
import datetime
from torch.autograd import Variable
import pdb
import sys
import torch.autograd as autograd
import torchvision.models as models
from torch.optim.lr_scheduler import LambdaLR
import random

from models.vae import *
from utils import AverageMeter, accuracy, setup_logger, AdamW
from dataset.randaugment import RandAugmentMC


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def reconst_images(batch_size=64, batch_num=1, dataloader=None, model=None):
    cifar10_dataloader = dataloader
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(cifar10_dataloader):
            if batch_idx >= batch_num:
                break
            else:
                X, y = X.cuda(), y.cuda().view(-1, )
                _, gx, _, _ = model(X)

                grid_X = torchvision.utils.make_grid(X[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_X.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X)]}, commit=False)
                grid_Xi = torchvision.utils.make_grid(gx[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_GX.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_Xi)]}, commit=False)
                grid_X_Xi = torchvision.utils.make_grid((X[:batch_size] - gx[:batch_size]).data, nrow=8, padding=2,
                                                        normalize=True)
                wandb.log({"_Batch_{batch}_RX.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X_Xi)]}, commit=False)
    print('reconstruction complete!')


def test(epoch, model, classifier, testloader):
    # set model as testing mode
    model.eval()
    classifier.eval()
    acc_gx_avg = AverageMeter()
    acc_rx_avg = AverageMeter()
    acc_class_avg = AverageMeter()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            # distribute data to device
            x, y = x.cuda(), y.cuda().view(-1, )
            bs = x.size(0)
            norm = torch.norm(torch.abs(x.view(bs, -1)), p=2, dim=1)
            _, gx, _, _ = model(x)
            out = classifier(x-gx)
            acc_gx = 1 - F.mse_loss(torch.div(gx, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / bs
            acc_rx = 1 - F.mse_loss(torch.div(x - gx, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / bs
            prec1, prec5 = accuracy(out, y, topk=(1, 5))
            acc_gx_avg.update(acc_gx.data.item(), bs)
            acc_rx_avg.update(acc_rx.data.item(), bs)
            acc_class_avg.update(prec1.item(), bs)

        wandb.log({'acc_gx_avg': acc_gx_avg.avg, \
                   'acc_rx_avg': acc_rx_avg.avg, \
                   'acc_class': acc_class_avg.avg}, commit=False)
        # plot progress
        print("\n| Validation Epoch #%d\t\tRec_gx: %.4f Rec_rx: %.4f ACC:  %.4f" % (epoch, acc_gx_avg.avg, acc_rx_avg.avg, acc_class_avg.avg))
        reconst_images(batch_size=64, batch_num=2, dataloader=testloader, model=model)
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
        print("Epoch {} model saved!".format(epoch + 1))


def run_batch(x, y, model, classifier, optimizer, re):

    x, y = x.cuda(), y.cuda().view(-1, )
    x, y = Variable(x), Variable(y)
    bs = x.size(0)

    _, gx,  mu, logvar = model(x)
    out = classifier(torch.cat((x-gx, gx, x), dim=0))
    out1 = out[0:x.size(0)]
    out2 = out[x.size(0):2*x.size(0)]
    out3 = out[2*x.size(0):]

    optimizer.zero_grad()
    l_rec = F.mse_loss(torch.zeros_like(gx), x-gx) # + F.l1_loss(torch.zeros_like(gx), gx)
    l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    l_kl /= bs * 3 * args.dim
    l_ce = F.cross_entropy(out1, y) + (F.softmax(out2, dim=1) * F.log_softmax(out2, dim=1)).sum(dim=1).mean() + F.cross_entropy(out3, y)
    loss = re * l_rec + args.kl * l_kl + args.ce * l_ce
    loss.backward()
    optimizer.step()

    return loss, l_rec, l_kl, l_ce


def main(args):
    setup_logger(args.save_dir)
    use_cuda = torch.cuda.is_available()
    print('\n[Phase 1] : Data Preparation')
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if (args.dataset == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        classifier = torchvision.models.resnet18(pretrained=False, num_classes=10)
    elif (args.dataset == 'cifar100'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
        classifier = torchvision.models.resnet18(pretrained=False, num_classes=100)
    else:
        print("No Such Dataset")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Model
    print('\n[Phase 2] : Model setup')
    model = CVAE_cifar_withbn(128, args.dim)

    if use_cuda:
        model.cuda()
        classifier.cuda()
        cudnn.benchmark = True

    optimizer = AdamW([
        {'params': model.parameters()},
        {'params': classifier.parameters()}
    ], lr=args.lr, betas=(0.9, 0.999), weight_decay=1.e-6)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.epochs)

    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(args.epochs))
    print('| Initial Learning Rate = ' + str(args.lr))

    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        classifier.train()

        loss_avg = AverageMeter()
        loss_rec = AverageMeter()
        loss_kl = AverageMeter()
        loss_ce = AverageMeter()

        if epoch < 100:
            re = args.re[0]
        elif epoch < 200:
            re = args.re[1]
        else:
            re = args.re[2]

        print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, optimizer.param_groups[0]['lr']))
        for batch_idx, (x, y) in enumerate(trainloader):
            bs = x.size(0)
            loss, l_rec, l_kl, l_ce = \
                run_batch(x, y, model, classifier, optimizer, re)
            loss_avg.update(loss.data.item(), bs)
            loss_rec.update(l_rec.data.item(), bs)
            loss_kl.update(l_kl.data.item(), bs)
            loss_ce.update(l_ce.data.item(), bs)

            n_iter = (epoch - 1) * len(trainloader) + batch_idx
            wandb.log({'loss': loss_avg.avg, \
                       'loss_rec': loss_rec.avg, \
                       'loss_kl': loss_kl.avg, \
                       'loss_ce': loss_ce.avg, \
                       'lr': optimizer.param_groups[0]['lr']}, step=n_iter)
            if (batch_idx + 1) % 30 == 0:
                sys.stdout.write('\r')
                sys.stdout.write(
                    '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\t Loss_rec: %.4f Loss_kl: %.4f Loss_ce: %.4f '
                    % (epoch, args.epochs, batch_idx + 1,
                       len(trainloader),  loss_rec.avg, loss_kl.avg, loss_ce.avg))
        scheduler.step()
        if epoch % 10 == 1:
            test(epoch, model, classifier, trainloader)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning_rate')
    parser.add_argument('--save_dir', default='./results/autoaug_new_8_0.5/', type=str, help='save_dir')
    parser.add_argument('--seed', default=666, type=int, help='seed')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
    parser.add_argument('--epochs', default=300, type=int, help='training_epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--dim', default=128, type=int, help='CNN_embed_dim')
    parser.add_argument('--re', nargs='+', type=int)
    parser.add_argument('--kl', default=1.0, type=float, help='kl weight')
    parser.add_argument('--ce', default=1.0, type=float, help='ce')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--clip_grad', type=float, default=0.1, help='clip gradients to this value')
    args = parser.parse_args()
    wandb.init(config=args, name=args.save_dir.replace("results/", ''))
    set_seed(args)

    main(args)
