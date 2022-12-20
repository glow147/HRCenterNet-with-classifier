import torch

def calc_loss(pred, class_pred, gt, metrics):

    mask = torch.sign(gt[..., 1])
    N = torch.sum(mask)

    _heatmap_loss = heatmap_loss(pred, gt, mask, metrics)
    _size_loss = size_loss(pred, gt, mask, metrics)
    _offset_loss = offset_loss(pred, gt, mask, metrics)
    _class_loss = class_loss(class_pred, gt, mask, metrics)

    all_loss = (-1 * _heatmap_loss + 10. * _size_loss + 5. * _offset_loss + _class_loss) / N
    classifier_loss = _class_loss / N

    metrics['loss'] = all_loss.detach().cpu().numpy()
    metrics['heatmap'] = (-1 *  _heatmap_loss / N).detach().cpu().numpy()
    metrics['size'] = (10. * _size_loss / N).detach().cpu().numpy()
    metrics['offset'] = (5. * _offset_loss / N).detach().cpu().numpy()
    metrics['class'] = (_class_loss / N).detach().cpu().numpy()

    return all_loss, classifier_loss

def class_loss(pred, gt, mask, metrics):

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    class_gt = torch.flatten(gt[..., 6]).long()
    pred = pred.permute(0, 2, 3, 1) # [batch_size, width, height, pred_logits]
    class_pred = torch.flatten(pred,start_dim=0, end_dim=2)
    classloss = criterion(class_pred, class_gt)

    return classloss


def heatmap_loss(pred, gt, mask, metrics):

    alpha = 2.
    beta = 4.

    heatmap_gt_rate = torch.flatten(gt[...,:1])
    heatmap_gt = torch.flatten(gt[...,1:2])
    heatmap_pred = torch.flatten(pred[:,:1,...])
    heatloss = torch.sum(heatmap_gt*((1-heatmap_pred)**alpha)*torch.log(heatmap_pred+1e-7) +
              (1-heatmap_gt)*((1-heatmap_gt_rate)**beta)*(heatmap_pred**alpha)*torch.log(1-heatmap_pred+1e-7))

    return heatloss

def offset_loss(pred, gt, mask, metrics):

    offsetloss = torch.sum(torch.abs(gt[...,2]*mask-pred[:,1,...]*mask)+torch.abs(gt[...,3]*mask-pred[:,2, ...]*mask))

    return offsetloss

def size_loss(pred, gt, mask, metrics):

    sizeloss = torch.sum(torch.abs(gt[...,4]*mask-pred[:,3, ...]*mask)+torch.abs(gt[...,5]*mask-pred[:,4,...]*mask))

    return sizeloss


