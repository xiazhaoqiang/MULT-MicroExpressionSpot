import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from lib.core.utils_af import iou_1d, jaccard_with_anchors, get_box_center, point_ifin_box, compute_max_iou_anchor2GT, \
    iou_simi, gather_topk_anchors


class ATSSAssigner(nn.Module):
    def __init__(self,
                 topk=9,
                 num_classes=80,
                 force_gt_matching=False,
                 eps=1e-9):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching

    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list,
                             pad_gt_mask):
        pad_gt_mask = pad_gt_mask.repeat(1, 1, self.topk).bool()
        gt2anchor_distances_list = torch.split(
            gt2anchor_distances, num_anchors_list, dim=-1)
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0, ] + num_anchors_index[:-1]
        is_in_topk_list = []        # [batch,50,num_anchors]
        topk_idxs_list = []
        for distances, anchors_index in zip(gt2anchor_distances_list,
                                            num_anchors_index):
            num_anchors = distances.shape[-1]
            topk_metrics, topk_idxs = torch.topk(
                distances, self.topk, axis=-1, largest=False)
            topk_idxs_list.append(topk_idxs + anchors_index)
            topk_idxs = torch.where(pad_gt_mask, topk_idxs,
                                    torch.zeros_like(topk_idxs))
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)

            is_in_topk = torch.where(is_in_topk > 1,
                                     torch.zeros_like(is_in_topk), is_in_topk)
            is_in_topk_list.append(is_in_topk.to(gt2anchor_distances.dtype))
        is_in_topk_list = torch.cat(is_in_topk_list, axis=-1)
        topk_idxs_list = torch.cat(topk_idxs_list, axis=-1)
        return is_in_topk_list, topk_idxs_list

    @torch.no_grad()
    def forward(self,
                anchor_bboxes,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None,
                pred_bboxes=None):
        assert gt_labels.ndim == gt_bboxes.ndim and gt_bboxes.ndim == 3

        num_anchors, _ = anchor_bboxes.shape
        batch_size, num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4])
            assigned_scores = torch.zeros(
                [batch_size, num_anchors, self.num_classes])
            return assigned_labels, assigned_bboxes, assigned_scores

        # 1. compute iou between gt and anchor bbox, [B, n, L]
        ious = iou_1d(anchor_bboxes,gt_bboxes.reshape([-1,2]))
        ious = ious.reshape([batch_size, -1, num_anchors])

        # 2. compute center distance between all anchors and gt, [B, n, L]
        gt_centers = get_box_center(gt_bboxes.reshape([-1, 2])).unsqueeze(1)       # [batch*max_action_num,1,1] --->[64*30,1,1]
        anchor_centers = get_box_center(anchor_bboxes)                             # [anchor_num,2] ---> [124,1]
        gt2anchor_distances = (gt_centers - anchor_centers.unsqueeze(0)) \
            .norm(2, dim=-1).reshape([batch_size, -1, num_anchors])

        # 3. on each pyramid level, selecting topk closest candidates
        # based on the center distance, [B, n, L]
        is_in_topk, topk_idxs = self._gather_topk_pyramid(
            gt2anchor_distances, num_anchors_list, pad_gt_mask)

        # 4. get corresponding iou for the these candidates, and compute the
        # mean and std, 5. set mean + std as the iou threshold
        iou_candidates = ious * is_in_topk

        # 计算iou_threshold
        temp_iou_candidates = iou_candidates.flatten(end_dim=-2)
        fe_num, num_anchors = temp_iou_candidates.shape
        temp_inds = topk_idxs.flatten(end_dim=-2) + (torch.arange(fe_num, device=topk_idxs.device) * num_anchors).unsqueeze(-1)
        iou_threshold = temp_iou_candidates.view(-1)[temp_inds]

        iou_threshold = iou_threshold.reshape([batch_size, num_max_boxes, -1])
        iou_threshold = iou_threshold.mean(axis=-1, keepdim=True) + \
                        iou_threshold.std(axis=-1, keepdim=True)
        is_in_topk = torch.where(
            iou_candidates > iou_threshold.repeat([1, 1, num_anchors]),
            is_in_topk, torch.zeros_like(is_in_topk))

        # 6. check the positive sample's center in gt, [B, n, L]
        is_in_gts = point_ifin_box(anchor_centers, gt_bboxes)

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # 7. if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        mask_positive_sum = mask_positive.sum(axis=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).repeat(
                [1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor2GT(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou,
                                        mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        # 8. make sure every gt_bbox matches the anchor
        """
        if self.force_gt_matching:
            is_max_iou = compute_max_iou_gt(ious) * pad_gt_mask
            mask_max_iou = (is_max_iou.sum(-2, keepdim=True) == 1).tile(
                [1, num_max_boxes, 1])
            mask_positive = torch.where(mask_max_iou, is_max_iou,
                                        mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        """
        assigned_gt_index = mask_positive.argmax(axis=-2)

        # assigned target
        batch_ind = torch.arange(
            end=batch_size, dtype=gt_labels.dtype, device=gt_labels.device).unsqueeze(-1)
        assigned_gt_index = (assigned_gt_index + batch_ind * num_max_boxes).long()
        assigned_labels = gt_labels.flatten()[assigned_gt_index.flatten()]
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(
            mask_positive_sum > 0, assigned_labels,
            torch.full_like(assigned_labels, bg_index))

        assigned_bboxes = gt_bboxes.reshape([-1,2])[assigned_gt_index.flatten()]
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 2])



        assigned_scores = F.one_hot(assigned_labels.long(), self.num_classes + 1).float()
        ind = list(range(self.num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = assigned_scores[:, :, :self.num_classes]
        # assigned_scores = torch.index_select(
        #     assigned_scores, torch.to_tensor(ind), axis=-1)
        if pred_bboxes is not None:
            # assigned iou
            # iou_similarity
            ious = iou_simi(gt_bboxes, pred_bboxes) * mask_positive
            ious = ious.max(axis=-2)[0].unsqueeze(-1)
            assigned_scores *= ious
        elif gt_scores is not None:
            raise NotImplementedError

        return assigned_labels.long(), assigned_bboxes, assigned_scores


class TaskAlignedAssigner(nn.Module):
    """TOOD: Task-aligned One-stage Object Detection
    """

    def __init__(self, topk=13, alpha=1.0, beta=6.0, eps=1e-9, num_classes=80):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(self,
                pred_scores,
                pred_bboxes,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py
        """
        assert pred_scores.ndim == pred_bboxes.ndim
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3

        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4])
            assigned_scores = torch.zeros(
                [batch_size, num_anchors, num_classes])
            return assigned_labels, assigned_bboxes, assigned_scores

        # compute iou between gt and pred bbox, [B, n, L]
        ious = iou_simi(gt_bboxes, pred_bboxes)
        # gather pred bboxes class score
        pred_scores = pred_scores.permute(0, 2, 1)      #[batch,num_classes,anchor_nums]
        gt_labels = gt_labels.long()
        batch_ind = torch.arange(
            end=batch_size, dtype=gt_labels.dtype, device=pred_scores.device).unsqueeze(-1)
        bbox_cls_scores = torch.zeros((batch_size, num_max_boxes, num_anchors), dtype=torch.float, device=pred_scores.device)
        for i in range(batch_size):
            bbox_cls_scores[i] = pred_scores[i, gt_labels[i].squeeze(-1)]
        # bbox_cls_scores = paddle.gather_nd(pred_scores, gt_labels_ind)
        # compute alignment metrics, [B, n, L]
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(
            self.beta)

        # check the positive sample's center in gt, [B, n, L]
        is_in_gts = point_ifin_box(anchor_points, gt_bboxes)

        # select topk largest alignment metrics pred bbox as candidates

        is_in_topk = gather_topk_anchors(
            alignment_metrics * is_in_gts,
            self.topk,
            topk_mask=pad_gt_mask.repeat([1, 1, self.topk]).to(torch.bool))
        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected, [B, n, L]
        mask_positive_sum = mask_positive.sum(axis=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).repeat(
                [1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor2GT(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        assigned_gt_index = mask_positive.argmax(axis=-2)
        # assigned target
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = gt_labels.flatten()[assigned_gt_index]
        assigned_labels = torch.where(
            mask_positive_sum > 0, assigned_labels,
            torch.full_like(assigned_labels, bg_index))

        assigned_bboxes = gt_bboxes.reshape([-1, 2])[assigned_gt_index]

        assigned_scores = F.one_hot(assigned_labels, num_classes + 1)
        ind = list(range(num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = assigned_scores[:, :, :bg_index]

        # rescale alignment metrics
        alignment_metrics *= mask_positive
        max_metrics_per_instance = alignment_metrics.max(axis=-1, keepdim=True)[0]
        max_ious_per_instance = (ious * mask_positive).max(axis=-1,
                                                           keepdim=True)[0]
        alignment_metrics = alignment_metrics / (
            max_metrics_per_instance + self.eps) * max_ious_per_instance
        alignment_metrics = alignment_metrics.max(-2)[0].unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        return assigned_labels, assigned_bboxes, assigned_scores


    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list,
                             pad_gt_mask):
        pad_gt_mask = pad_gt_mask.repeat(1, 1, self.topk).bool()
        gt2anchor_distances_list = torch.split(
            gt2anchor_distances, num_anchors_list, dim=-1)
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0, ] + num_anchors_index[:-1]
        is_in_topk_list = []
        topk_idxs_list = []
        for distances, anchors_index in zip(gt2anchor_distances_list,
                                            num_anchors_index):
            num_anchors = distances.shape[-1]
            topk_metrics, topk_idxs = torch.topk(
                distances, self.topk, axis=-1, largest=False)
            topk_idxs_list.append(topk_idxs + anchors_index)
            topk_idxs = torch.where(pad_gt_mask, topk_idxs,
                                    torch.zeros_like(topk_idxs))
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)

            is_in_topk = torch.where(is_in_topk > 1,
                                     torch.zeros_like(is_in_topk), is_in_topk)
            is_in_topk_list.append(is_in_topk.to(gt2anchor_distances.dtype))
        is_in_topk_list = torch.cat(is_in_topk_list, axis=-1)
        topk_idxs_list = torch.cat(topk_idxs_list, axis=-1)
        return is_in_topk_list, topk_idxs_list