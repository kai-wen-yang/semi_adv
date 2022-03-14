import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
import wandb
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model
from tqdm import tqdm
import torch.distributed as dist
from dataset.cifar_index import DATASET_GETTERS, mu_cifar100, std_cifar100, clamp
from utils import AverageMeter, accuracy, setup_logger

import models
import pdb
import math
import torchvision
from torch.autograd import Variable
from typing import List, Optional, Tuple, Union, cast

logger = logging.getLogger(__name__)
best_acc = 0


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def reconst_images(x_adv, strong_x, run):
    grid_X = torchvision.utils.make_grid(strong_x[:10].data, nrow=10, padding=2, normalize=True)
    grid_AdvX = torchvision.utils.make_grid(x_adv[:10].data, nrow=10, padding=2, normalize=True)
    grid_Delta = torchvision.utils.make_grid(x_adv[:10]-strong_x[:10].data, nrow=10, padding=2, normalize=True)
    grid = torch.cat((grid_X, grid_AdvX, grid_Delta), dim=1)
    run.log({"Batch.jpg": [
        wandb.Image(grid)]}, commit=False)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def normalize_flatten_features(
    features: Tuple[torch.Tensor, ...],
    eps=1e-10,
) -> torch.Tensor:

    normalized_features: List[torch.Tensor] = []
    for feature_layer in features:
        norm_factor = torch.sqrt(
            torch.sum(feature_layer ** 2, dim=1, keepdim=True)) + eps
        normalized_features.append(
            (feature_layer / (norm_factor *
                              np.sqrt(feature_layer.size()[2] *
                                      feature_layer.size()[3])))
            .view(feature_layer.size()[0], -1)
        )
    return torch.cat(normalized_features, dim=1)


def normalize_features(
    features,
    eps=1e-10,
):
    norm_factor = torch.sqrt(torch.sum(features ** 2, dim=1, keepdim=True)) + eps
    return (features / (norm_factor * np.sqrt(features.size()[2] *features.size()[3]))).view(features.size()[0], -1)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


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


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='stl10', type=str,
                        choices=['stl10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--dim', default=512, type=int, help='CNN_embed_dim')
    parser.add_argument('--T_adv', default=5, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--warmup_adv', default=5, type=int, help='warm up epoch')
    parser.add_argument('--attack-iters', default=7, type=int, help='Attack iterations')
    parser.add_argument('--step', default=0.02, type=float, help='eps for adversarial')
    parser.add_argument('--ce', default=1, type=float, help='eps for adversarial')
    args = parser.parse_args()
    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as nets
            model = nets.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes,
                                            bn_adv_flag=True,
                                            bn_adv_momentum=0.01)
        elif args.arch == 'resnext':
            import models.resnext as nets
            model = nets.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        print("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    print(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
    )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        run = wandb.init(
            config=args, name=args.out, save_code=True,
        )
        setup_logger(args.out)
    if args.dataset == 'stl10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 34
            args.model_width = 2
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, '../data')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)
    upper_limit = ((1 - mu_cifar100) / std_cifar100).to(args.device)
    lower_limit = ((0 - mu_cifar100) / std_cifar100).to(args.device)
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)
    ### init eps bank
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        print("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = DDP(model, delay_allreduce=True)

    print("***** Running training *****")
    print(f"  Task = {args.dataset}@{args.num_labeled}")
    print(f"  Num Epochs = {args.epochs}")
    print(f"  Batch size per GPU = {args.batch_size}")
    print(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    print(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    if args.amp:
        from apex import amp
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x_w, targets_x, _ = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x_w, targets_x, _ = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), targets_ux, index = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), targets_ux, index = unlabeled_iter.next()

            data_time.update(time.time() - end)
            model.train()
            all_x = torch.cat((inputs_x_w, inputs_u_w, inputs_u_s)).to(args.device)
            targets_x = targets_x.to(args.device)
            targets_ux = targets_ux.to(args.device)
            inputs_u_w = inputs_u_w.to(args.device)
            batch_size = inputs_x_w.size(0)

            optimizer.zero_grad()

            logits, _ = model(all_x, return_feature=True)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            l_ce = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            mask_index = max_probs.ge(args.threshold)
            ce_s = F.cross_entropy(logits_u_s, targets_u, reduction='none')
            l_cs = (ce_s * mask).mean()

            input_selected = inputs_u_w[mask_index]
            targets_selected = targets_u[mask_index]

            if args.world_size > 1:
                mask_all = torch.cat(GatherLayer.apply(mask), dim=0)
                mask1, mask2, mask3, mask4 = mask_all.chunk(4)
            else:
                mask1 = mask
                mask2 = mask
                mask3 = mask
                mask4 = mask
            if mask1.sum() == 0 or mask2.sum() == 0 or mask3.sum() == 0 or mask4.sum() == 0 or epoch <= args.warmup_adv:
                loss = l_ce + l_cs
                with torch.no_grad():
                    prec, _ = accuracy(logits_x.data, targets_x.data, topk=(1, 5))
                    prec_unlab, _ = accuracy(logits_u_w.data, targets_ux.data, topk=(1, 5))
                    prec_unlab_strong, _ = accuracy(logits_u_s.data, targets_ux.data, topk=(1, 5))

                    if args.local_rank in [-1, 0]:
                        run.log({'loss/l_cs': l_cs.data.item(),
                                 'loss/l_ce': l_ce.data.item(),
                                 'ACC/acc': prec.item(),
                                 'ACC/acc_unlab': prec_unlab.item(),
                                 'ACC/acc_unlab_strongaug': prec_unlab_strong.item(),
                                 'mask': mask.mean().item(),
                                 'lr': optimizer.param_groups[0]['lr']})
            else:
                ##CDAA
                #random init
                eps = args.step # bs
                alpha = eps / 4
                delta = torch.zeros_like(input_selected.detach()).to(args.device)
                delta.uniform_(-1, 1) # bs, 3, 32, 32
                delta = delta * eps
                delta.data = clamp(delta, lower_limit - input_selected, upper_limit - input_selected)
                delta.requires_grad = True

                with torch.no_grad():
                    logits_selected, feat_selected = model(input_selected, return_feature=True)
                    y_w = torch.gather(torch.softmax(logits_selected, dim=-1), 1, targets_selected.view(-1, 1)).squeeze(dim=1)
                for _ in range(args.attack_iters):
                    logits_adv, feat_adv = model(input_selected + delta, adv=True, return_feature=True)
                    pip = (normalize_flatten_features(feat_adv) - normalize_flatten_features(feat_selected).detach()).norm(dim=1).mean()
                    ce = F.cross_entropy(logits_adv, targets_selected)
                    loss_tmp = pip - args.ce * ce
                    if args.amp:
                        with amp.scale_loss(loss_tmp, optimizer) as scaled_loss:
                            scaled_loss.backward(retain_graph=True)
                    else:
                        loss_tmp.backward(retain_graph=True)

                    grad = delta.grad.detach()
                    delta.data = torch.clamp(delta + alpha * torch.sign(grad), -eps, eps)
                    delta.data = clamp(delta, lower_limit - input_selected, upper_limit - input_selected)
                    delta.grad.zero_()
                delta = delta.detach()
                ##
                logits_adv, feat_adv = model(input_selected + delta, adv=True, return_feature=True)
                _, targets_adv = torch.max(logits_selected, 1)
                y_adv = torch.gather(torch.softmax(logits_adv, dim=-1), 1, targets_selected.view(-1, 1)).squeeze(dim=1)
                #####
                l_adv = F.cross_entropy(logits_adv, targets_selected)
                loss = l_ce + l_cs + l_adv

                with torch.no_grad():
                    prec, _ = accuracy(logits_x.data, targets_x.data, topk=(1, 5))
                    prec_unlab, _ = accuracy(logits_u_w.data, targets_ux.data, topk=(1, 5))
                    prec_unlab_strong, _ = accuracy(logits_u_s.data, targets_ux.data, topk=(1, 5))

                    prec_pesudo_label = (targets_u == targets_ux).float()[max_probs.ge(args.threshold)].mean()
                    prec_pesudo_adv = (targets_selected == targets_adv).float().mean()
                    if args.local_rank in [-1, 0]:
                        run.log({'loss/l_cs': l_cs.data.item(),
                                 'loss/l_ce': l_ce.data.item(),
                                 'loss/l_adv': l_adv.data.item(),
                                 'Adv/y_w': y_w.mean().data.item(),
                                 'Adv/y_adv': y_adv.mean().data.item(),
                                 'Adv/epsilon_mean_selected': eps.mean().item(),
                                 'His/epsilon_selected': wandb.Histogram(eps.cpu().detach().numpy(), num_bins=512),
                                 'His/y_w': wandb.Histogram(y_w.cpu().detach().numpy(), num_bins=512),
                                 'His/y_adv': wandb.Histogram(y_adv.cpu().detach().numpy(), num_bins=512),
                                 'His/y_delta': wandb.Histogram((y_adv-y_w).cpu().detach().numpy(), num_bins=512),
                                 'ACC/acc': prec.item(),
                                 'ACC/acc_unlab': prec_unlab.item(),
                                 'ACC/acc_unlab_strongaug': prec_unlab_strong.item(),
                                 'pesudo/prec_label': prec_pesudo_label.item(),
                                 'pesudo/prec_adv': prec_pesudo_adv.item(),
                                 'mask': mask.mean().item(),
                                 'lr': optimizer.param_groups[0]['lr']})
                        if batch_idx == 1:
                            reconst_images(input_selected + delta, input_selected, run)

            optimizer.zero_grad()
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            losses_x.update(l_ce.item())
            losses_u.update(l_cs.item())
            mask_probs.update(mask.mean().item())
            scheduler.step()

            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s.  Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}.  Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)
            run.log({'test/1.test_acc': test_acc,
                         'test/2.test_loss': test_loss})

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            print('Best top-1 acc: {:.2f}'.format(best_acc))
            print('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        run.finish()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader, disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    print("top-1 acc: {:.2f}".format(top1.avg))
    print("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()