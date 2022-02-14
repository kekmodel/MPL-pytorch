import logging
import os
import shutil
from collections import OrderedDict

import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def create_loss_fn(args):
    # if args.label_smoothing > 0:
    #     criterion = SmoothCrossEntropyV2(alpha=args.label_smoothing)
    # else:  
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    return criterion.to(args.device)


def module_load_state_dict(model, state_dict):
    try:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = f'module.{k}'  # add `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def model_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        module_load_state_dict(model, state_dict)


def save_checkpoint(args, state, is_best, finetune=False):
    os.makedirs(args.save_path, exist_ok=True)
    if finetune:
        name = f'{args.name}_finetune'
    else:
        name = args.name
    filename = f'{args.save_path}/{name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    if is_best:
        shutil.copyfile(filename, f'{args.save_path}/{args.name}_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    output = output.to(torch.device('cpu'))
    target = target.to(torch.device('cpu'))
    maxk = max(topk)
    batch_size = target.shape[0]

    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, maxk).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        if self.alpha == 0:
            loss = F.cross_entropy(logits, labels)
        else:
            num_classes = logits.shape[-1]
            alpha_div_k = self.alpha / num_classes
            target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
                (1. - self.alpha) + alpha_div_k
            loss = (-(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)).mean()
        return loss


class SmoothCrossEntropyV2(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, label_smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__()
        assert label_smoothing < 1.0
        self.smoothing = label_smoothing
        self.confidence = 1. - label_smoothing

    def forward(self, x, target):
        if self.smoothing == 0:
            loss = F.cross_entropy(x, target)
        else:
            logprobs = F.log_softmax(x, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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
