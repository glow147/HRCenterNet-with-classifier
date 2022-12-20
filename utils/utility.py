import torchvision
import torch
import yaml
import math
from easydict import EasyDict
from torch.optim.lr_scheduler import _LRScheduler

def load_setting(setting):

    with open(setting, 'r', encoding='utf8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return EasyDict(cfg)

def get_score(predict, class_predict, label, size, output_size, nms_score, iou_threshold, iou_criterion):
    bbox = list()
    score_list = list()
    heatmap = predict[0, ...]
    offset_y = predict[1, ...]
    offset_x = predict[2, ...]
    width_map = predict[3, ...]
    height_map = predict[4, ...]
    class_map = torch.argmax(class_predict, dim=0)

    gt_bbox = label[..., 1:]
    gt_labels = label[..., 0]
    x_c, y_c, w, h = gt_bbox.unbind(-1)

    b = [(x_c - 0.5*w), (y_c - 0.5*h),
         (x_c + 0.5*w), (y_c + 0.5*h)]
    gt_bbox = torch.stack(b, dim=-1)

    gt_img_size = size

    heatmap_indices = torch.where(heatmap.reshape(-1,1) >= nms_score)[0]

    if len(heatmap_indices) == 0:
        return 0.,0.,0.

    rows = heatmap_indices // output_size[1]
    cols = heatmap_indices - rows*output_size[0]

    bias_x = offset_x[rows, cols] * (gt_img_size[1] / output_size[1])
    bias_y = offset_y[rows, cols] * (gt_img_size[0] / output_size[0])

    width = width_map[rows, cols] * output_size[1] * (gt_img_size[1] / output_size[1])
    height = height_map[rows, cols] * output_size[0] * (gt_img_size[0] / output_size[0])

    score_list = heatmap[rows, cols]
    preds = class_map[rows, cols]

    rows = rows * (gt_img_size[1] / output_size[1]) + bias_y
    cols = cols * (gt_img_size[0] / output_size[0]) + bias_x

    lys = rows - width // 2
    lxs = cols - height // 2
    rys = rows + width // 2
    rxs = cols + height // 2

    bbox = torch.stack((lxs,lys,rxs,rys),dim=1)

    if len(bbox) == 0:
        return 0., 0., 0.

    _nms_index = torchvision.ops.nms(bbox, scores=torch.flatten(score_list), iou_threshold=iou_threshold)

    _nms_bbox, _nms_preds = bbox[_nms_index], preds[_nms_index]
    result = torchvision.ops.box_iou(_nms_bbox, gt_bbox)
    best_ious, indices = torch.max(result, dim=1)

    matching_labels = gt_labels[indices]
    # calc recall
    recall_tp = torch.sum(best_ious >= iou_criterion).item()

    # calc precision
    pred_labels = _nms_preds[best_ious >= iou_criterion]
    precision_tp = torch.sum( pred_labels == matching_labels[best_ious >= iou_criterion] ).item()

    # calc f1_score
    precision = precision_tp / (len(pred_labels)+1e-7)
    recall = recall_tp / len(gt_bbox)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return precision, recall, f1

class CustomCosineAnnealingWarmupRestarts(_LRScheduler):
    """
        src: https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py

        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle

        super(CustomCosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
