import os
import time
import json
import shutil
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from dataset_3d import TSNDataSet_3D
from transforms import *
from opts import parser
from ops.models import TSN

from sklearn.metrics import confusion_matrix

best_prec1 = 0


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main():
    global args, best_prec1
    args = parser.parse_args()

    if (args.dataset == 'something') or (args.dataset == 'something_v2'):
        args.num_class = 174
    elif args.dataset == 'diving':
        args.num_class = 48
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    img_tmpl = "{:05d}.jpg"
    if args.dataset == 'something':
        if args.modality == 'RGB':
            img_prefix = ''
            img_tmpl = "{:05d}.jpg"
    elif args.dataset == 'something_v2':
        if args.modality == 'RGB':
            img_prefix = ''
            img_tmpl = "{:06d}.jpg"
        else:
            img_prefix = ''
            img_tmpl = "{:06d}.jpg"
    elif args.dataset == 'diving':
        if args.modality == 'RGB':
            img_prefix = 'frames'
            img_tmpl = "{:05d}.jpg"
        else:
            img_prefix = ''
            img_tmpl = "{:06d}.jpg"
    else:
        if args.modality == 'RGB':
            img_prefix = 'img_'
        else:
            img_prefix = 'flow_{}_'

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    with open(os.path.join(args.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(args), opt_file)
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    totensor_div = True

    def count_parameters(model):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params / 1000000

    # setup the model
    model = TSN(args.num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool, cttlstm=args.cttlstm,short_len=args.short_len,
                long_len=args.long_len, new_length=args.new_length, dataset=args.dataset)
    num_params = count_parameters(model)
    print('total param is ', num_params)

    model.cuda()
    model = nn.DataParallel(model)

    train_augmentation = torchvision.transforms.Compose([
        GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66],
                            max_distort=1,
                            fix_crop=False)
    ])

    policies = model.module.get_optim_policies()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            checkpoint_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            checkpoint_dict = {
                k: v for k, v in checkpoint_dict.items() if k in model_dict
            }
            model_dict.update(checkpoint_dict)
            model.load_state_dict(model_dict)
            print(("=> loaded checkpoint '{}' (epoch {})".format(
                args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    cudnn.benchmark = True

    normalize = GroupNormalize(input_mean, input_std)
    rev_normalize = ReverseGroupNormalize(input_mean, input_std)

    if not args.evaluate:
        train_temp_transform = IdentityTransform()
        train_loader = torch.utils.data.DataLoader(
            TSNDataSet_3D('',
                          args.train_list,
                          num_segments_lst=[args.num_segments],
                          modality=args.modality,
                          new_length=args.new_length,
                          image_tmpl=img_prefix + img_tmpl,
                          temp_transform=train_temp_transform,
                          transform=torchvision.transforms.Compose([
                              train_augmentation,
                              Stack(roll=args.arch == 'BNInception'),
                              ToTorchFormatTensor(div=totensor_div),
                              normalize,
                          ]),
                          gap=args.gap,
                          dataset=args.dataset,
                          dense_sample=args.dense_sample),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn)

    val_temp_transform = IdentityTransform()

    args.n_cframes_lst = [args.num_segments]

    cropping = torchvision.transforms.Compose([
        GroupScale(args.input_size),
        GroupCenterCrop((args.input_size, int(1.143 * args.input_size)))
    ])

    val_loader = torch.utils.data.DataLoader(TSNDataSet_3D(
        '',
        args.val_list,
        num_segments_lst=args.n_cframes_lst,
        new_length=args.new_length,
        modality=args.modality,
        image_tmpl=img_prefix + img_tmpl,
        random_shift=False,
        temp_transform=val_temp_transform,
        transform=torchvision.transforms.Compose([
            cropping,
            Stack(roll=args.arch == 'BNInception'),
            ToTorchFormatTensor(div=totensor_div),
            normalize,
        ]),
        gap=args.gap,
        dataset=args.dataset,
        dense_sample=args.dense_sample),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    ce_criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))


    if args.evaluate:
        print('evaluating')
        val_logger = os.path.join(args.result_path, 'evaluate.log')
        validate(val_loader,
                 model,
                 ce_criterion,
                 0,
                 val_logger=val_logger,
                 rev_normalize=rev_normalize,
                 epoch=args.start_epoch,
                 last_result_json=args.last_result_json,
                 this_result_json=args.this_result_json)
        return

    train_logger = os.path.join(args.result_path, 'train.log')
    val_logger = os.path.join(args.result_path, 'val.log')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader,
              model,
              ce_criterion,
              optimizer,
              epoch,
              train_logger=train_logger)
        with open(train_logger, 'a') as f:
            f.write('\n')

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            },
            False,
            filename='ep_' + str(epoch) + '_checkpoint.pth.tar')

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader,
                             model,
                             ce_criterion, (epoch + 1) * len(train_loader),
                             val_logger=val_logger,
                             rev_normalize=rev_normalize,
                             epoch=epoch)

            # remember best prec@1 and save checkpoint
            if prec1 > best_prec1:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': 0,
                    }, is_best)


def train(train_loader, model, ce_criterion, optimizer, epoch, train_logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    copy_gpu_time = AverageMeter()
    calc_loss_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    step_time = AverageMeter()
    ce_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    np.random.seed()
    end = time.time()
    for i, (_, inputs, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        start_copy_gpu = time.time()

        target = target.cuda()
        input_var = torch.autograd.Variable(inputs).cuda()
        target_var = torch.autograd.Variable(target)

        input_size = inputs.size()

        end_copy_gpu = time.time()
        copy_gpu_time.update(end_copy_gpu - start_copy_gpu)

        start_forwarding = time.time()
        # compute output
        output = model(input_var)
        end_forwarding = time.time()
        forward_time.update(end_forwarding - start_forwarding)

        start_calc_loss = time.time()
        loss = ce_criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))
        end_calc_loss = time.time()
        calc_loss_time.update(end_calc_loss - start_calc_loss)

        start_backwarding = time.time()
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(
                    total_norm, args.clip_gradient / total_norm))

        end_backwarding = time.time()
        backward_time.update(end_backwarding - start_backwarding)

        start_stepping = time.time()
        optimizer.step()
        end_stepping = time.time()
        step_time.update(end_stepping - start_stepping)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i > 44:
        # break

        if i % args.print_freq == 0:
            log_line = (
                'Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Copy_GPU {copy_gpu_time.val:.3f} ({copy_gpu_time.avg:.3f})\t'
                'Calc_Loss {calc_loss_time.val:.3f} ({calc_loss_time.avg:.3f})\t'
                'Forward {forward_time.val:.7f} ({forward_time.avg:.7f})\t'
                'Backward {backward_time.val:.7f} ({backward_time.avg:.7f})\t'
                'Step {step_time.val:.7f} ({step_time.avg:.7f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    copy_gpu_time=copy_gpu_time,
                    calc_loss_time=calc_loss_time,
                    forward_time=forward_time,
                    backward_time=backward_time,
                    step_time=step_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                    lr=optimizer.param_groups[-1]['lr'] * 0.1))
            print(log_line)
            with open(train_logger, 'a') as f:
                f.write(log_line + '\n')
        if (i + 1) % 2000 == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                },
                False,
                filename='ep_' + str(epoch) + 'iter_' + str(i) +
                         '_checkpoint.pth.tar')


def validate(val_loader,
             model,
             ce_criterion,
             iter,
             val_logger,
             rev_normalize,
             epoch,
             last_result_json=None,
             this_result_json=None):
    if last_result_json:
        with open(last_result_json, 'r') as f:
            last_result_dict = json.load(f)
        with open(this_result_json, 'w') as f:
            f.write(' ')
        this_result_dict = last_result_dict
    else:
        this_result_dict = {}
        if this_result_json:
            with open(this_result_json, 'w') as f:
                f.write(' ')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    copy_gpu_time = AverageMeter()
    calc_loss_time = AverageMeter()
    forward_time = AverageMeter()
    losses = AverageMeter()
    ce_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    scores = []
    labels = []
    for i, (vid_names, inputs, target, _) in enumerate(val_loader):
        if i == 200:
            batch_time.reset()
            data_time.reset()
            copy_gpu_time.reset()
            calc_loss_time.reset()
            forward_time.reset()

        data_time.update(time.time() - end)
        start_copy_gpu = time.time()
        target = target.cuda()
        input_var = torch.autograd.Variable(inputs, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

        input_size = inputs.size()
        end_copy_gpu = time.time()
        copy_gpu_time.update(end_copy_gpu - start_copy_gpu)

        torch.cuda.synchronize()
        start_forwarding = time.time()
        with torch.no_grad():
            output = (model(input_var))
        torch.cuda.synchronize()
        end_forwarding = time.time()
        forward_time.update(end_forwarding - start_forwarding)

        start_calc_loss = time.time()


        loss = ce_criterion(output, target_var)

        output = torch.softmax(output, -1)

        if args.evaluate:
            output_np = output.detach().cpu().data.numpy()
            output_fusion = []

            for j, vid_name in enumerate(vid_names):
                if last_result_json:
                    this_result_dict[vid_name] += args.this_model_weight * output_np[j]
                    output_fusion.append(this_result_dict[vid_name])
                else:
                    this_result_dict[vid_name] = output_np[j]
            scores.append(output.data.cpu().numpy().copy())
            labels.append(target.data.cpu().numpy().copy())
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        # if i > 44:
        # break
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))
        end_calc_loss = time.time()
        calc_loss_time.update(end_calc_loss - start_calc_loss)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log_line = (
                'Test: Epoch:[{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Copy_GPU {copy_gpu_time.val:.3f} ({copy_gpu_time.avg:.3f})\t'
                'Calc_Loss {calc_loss_time.val:.3f} ({calc_loss_time.avg:.3f})\t'
                'Forward {forward_time.val:.7f} ({forward_time.avg:.7f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch,
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    copy_gpu_time=copy_gpu_time,
                    calc_loss_time=calc_loss_time,
                    forward_time=forward_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5))
            print(log_line)
            with open(val_logger, 'a') as f:
                f.write(log_line + '\n')

    log_line = ('Testing Results: Prec@1 {top1.avg:.3f} '
                'Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'.format(top1=top1,
                                                                   top5=top5,
                                                                   loss=losses))

    print(log_line)
    with open(val_logger, 'a') as f:
        f.write(log_line + '\n\n')
    if args.evaluate:
        if this_result_json:
            with open(this_result_json, 'w') as f:
                json.dump(this_result_dict, f, cls=MyEncoder)
        video_pred = np.concatenate([np.argmax(x, axis=1) for x in scores])
        video_labels = np.concatenate([x for x in labels])
        cf = confusion_matrix(video_labels, video_pred).astype(float)
        np.save('cm_sthv1_8f.npy', cf)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        np.save('cls_acc', cls_acc)
        print(cls_acc)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    print('saving checkpoint...')
    filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    torch.save(state, os.path.join(args.result_path, filename))
    if is_best:
        print('it is also the best checkpoint...')
        best_name = '_'.join(
            (args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        with open(os.path.join(args.result_path, 'best_epoch.txt'), 'a') as f:
            f.write('best epoch: ' + str(state['epoch']))
        shutil.copyfile(os.path.join(args.result_path, filename),
                        os.path.join(args.result_path, best_name))


class MyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
