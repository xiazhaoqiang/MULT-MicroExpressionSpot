import torch
import torch.nn as nn
import torch.nn.functional as F

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()

class VarifocalLoss(nn.Module):

    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self,
                pred_score,
                gt_score,
                label,
                alpha=0.75,
                gamma=2.0):

        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = (F.binary_cross_entropy(pred_score, gt_score, reduction='none') * weight).sum()

        return loss

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy(
            score, label, weight=weight, reduction='sum')
        return loss


class BboxLoss(nn.Module):
    def __init__(self, num_classes, reg_max):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

    def forward(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).repeat([1, 1, 2])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                  bbox_mask).reshape([-1, 2])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 2])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).repeat(
                [1, 1, (self.reg_max + 1) * 2])
            pred_dist_pos = torch.masked_select(
                pred_dist, dist_mask).reshape([-1, 2, self.reg_max + 1])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 2])
            loss_dfl = self._df_loss(pred_dist_pos,
                                     assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = torch.zeros([1])
            loss_iou = torch.zeros([1])
            loss_dfl = torch.zeros([1])

        return loss_l1, loss_iou, loss_dfl

    def _bbox2distance(self, points, bbox):
        left, right = torch.split(bbox, 1, -1)
        ld = points - left
        rd = right - points
        return torch.cat([ld, rd], -1).clip(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

def abs_smooth(x):
    '''
    Sommth L1 loss
    Defined as:
        x^2 / 2        if abs(x) < 1
        abs(x) - 0.5   if abs(x) >= 1
    '''
    absx = torch.abs(x)
    minx = torch.min(absx, torch.tensor(1.0).type_as(dtype))
    loss = 0.5 * ((absx - 1) * minx + absx)
    loss = loss.mean()
    return loss


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes, device=labels.device)
    return y[labels]


def sel_fore_reg(cls_label_view, target_regs, pred_regs):
    '''
    Args:
        cls_label_view: bs*sum_t
        target_regs: bs, sum_t, 1
        pred_regs: bs, sum_t, 1
    Returns:
    '''
    sel_mask = cls_label_view >= 1.0
    target_regs_view = target_regs.view(-1)
    target_regs_sel = target_regs_view[sel_mask]
    pred_regs_view = pred_regs.view(-1)
    pred_regs_sel = pred_regs_view[sel_mask]

    return target_regs_sel, pred_regs_sel


def iou_loss(pred, target):
    inter_min = torch.max(pred[:, 0], target[:, 0])
    inter_max = torch.min(pred[:, 1], target[:, 1])
    inter_len = (inter_max - inter_min).clamp(min=0)
    union_len = (pred[:, 1] - pred[:, 0]) + (target[:, 1] - target[:, 0]) - inter_len
    tious = inter_len / union_len
    loss = (1 - tious).mean()
    return loss

