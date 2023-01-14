import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()



def batch_distance2bbox(points, distance, max_shapes=None):
    lt, rb = torch.split(distance, 1, -1)
    # while tensor add parameters, parameters should be better placed on the second place
    left = -lt + points
    right = rb + points
    out_bbox = torch.cat([left, right], -1)
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.ndim - max_shapes.ndim
        for _ in range(delta_dim):
            max_shapes.unsqueeze_(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox,
                               torch.zeros_like(out_bbox))
    return out_bbox


def iou_1d(pred, target):
    res_iou = []
    for  i in range(target.shape[0]):
        tmp_target = target[i].repeat(pred.shape[0]).reshape([-1,2])
        inter_min = torch.max(pred[:, 0], tmp_target[:,0])
        inter_max = torch.min(pred[:, 1], tmp_target[:,1])
        inter_len = (inter_max - inter_min).clamp(min=0)
        union_len = (pred[:, 1] - pred[:, 0]) + (tmp_target[:,1] - tmp_target[:,0]) - inter_len
        tious = inter_len / union_len
        res_iou.append(tious[None])
    res_iou = torch.cat(res_iou,dim=0)
    return res_iou

def get_box_center(boxes):
    """
        boxes: [n,2]
    """
    center = (boxes[:,0] + boxes[:,1])/2
    center = center[...,None]
    return center

def point_ifin_box(points,bboxes,eps=1e-9):
    r"""
    Args:
        points (Tensor, float32): shape[L, 1], center
        bboxes (Tensor, float32): shape[B, n, 2], "left,right"
        eps (float): Default: 1e-9
    Returns:
        is_in_bboxes (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    center = points.unsqueeze(0).unsqueeze(0)       #[1,1,L,1]
    left,rigth= bboxes.unsqueeze(2).chunk(2, axis=-1)
    l = center - left
    r = rigth - center
    bbox_ltrb = torch.cat([l, r], axis=-1)
    return (bbox_ltrb.min(axis=-1)[0] > eps).to(bboxes.dtype)


def compute_max_iou_anchor2GT(ious):
    num_max_boxes = ious.shape[-2]
    max_iou_index = ious.argmax(axis=-2)
    is_max_iou = F.one_hot(max_iou_index, num_max_boxes).permute(0, 2, 1)
    # return is_max_iou.to(ious.dtype)
    return is_max_iou.to(torch.float32)


def iou_simi(box1, box2, eps=1e-9):
    """Calculate iou of box1 and box2
    Args:
        box1 (Tensor): box with the shape [batch, max_ac_num, 2]
        box2 (Tensor): box with the shape [batch, anchor_sum, 2]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [batch, max_ac_num, anchor_sum]
    """
    box1 = box1.unsqueeze(2)  # [batch, max_ac_num, 2] -> [batch, max_ac_num, 1,2]
    box2 = box2.unsqueeze(1)  # [batch, anchor_sum, 2] -> [batch,1, anchor_sum, 2]
    gt_left,gt_right = box1[:,:,:,0],box1[:,:,:,1]
    pred_left,pred_right = box2[:,:,:,0],box2[:,:,:,1]
    left = torch.maximum(pred_left, gt_left)
    right = torch.minimum(pred_right, gt_right)
    overlap = (right - left).clip(0)
    area1 = (pred_right - pred_left).clip(0)
    area2 = (gt_right - gt_left).clip(0)
    union = area1 + area2 - overlap + eps
    return overlap / union

def jaccard_with_anchors(anchors_min, anchors_max, box_min, box_max):
    '''
    calculate tIoU for anchors and ground truth action segment
    '''
    inter_xmin = torch.max(anchors_min, box_min)
    inter_xmax = torch.min(anchors_max, box_max)
    inter_len = inter_xmax - inter_xmin

    inter_len = torch.max(inter_len, torch.tensor(0.0).type_as(dtype))
    union_len = anchors_max - anchors_min - inter_len + box_max - box_min

    jaccard = inter_len / union_len
    return jaccard


def tiou(anchors_min, anchors_max, len_anchors, box_min, box_max):
    '''
    calculate jaccatd score between a box and an anchor
    '''
    inter_xmin = np.maximum(anchors_min, box_min)
    inter_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(inter_xmax-inter_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    tiou = np.divide(inter_len, union_len)
    return tiou

def gather_topk_anchors(metrics, topk, largest=True, topk_mask=None, eps=1e-9):
    r"""
    Args:
        metrics (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
        topk (int): The number of top elements to look for along the axis.
        largest (bool) : largest is a flag, if set to true,
            algorithm will sort by descending order, otherwise sort by
            ascending order. Default: True
        topk_mask (Tensor, bool|None): shape[B, n, topk], ignore bbox mask,
            Default: None
        eps (float): Default: 1e-9
    Returns:
        is_in_topk (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = torch.topk(
        metrics, topk, axis=-1, largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > eps).tile(
            [1, 1, topk])
    topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
    is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
    is_in_topk = torch.where(is_in_topk > 1,
                              torch.zeros_like(is_in_topk), is_in_topk)
    return is_in_topk.to(metrics.dtype)





# Notice: maybe merge this with result_process_ab

def result_process_af(video_names, start_frames, cls_scores, anchors_xmin, anchors_xmax, cfg):
    # anchors_class,... : bs, sum_i(t_i*n_box), n_class
    # anchors_xmin, anchors_xmax: bs, sum_i(t_i*n_box)
    # video_names, start_frames: bs,
    out_df = pd.DataFrame()

    # feat_tem_width = cfg.MODEL.TEMPORAL_LENGTH[0]
    frame_window_width = cfg.DATASET.WINDOW_SIZE

    xmins = np.maximum(anchors_xmin, 0)
    xmins = xmins + np.expand_dims(start_frames, axis=1)
    # xmins = xmins / feat_tem_width * frame_window_width + np.expand_dims(start_frames, axis=1)
    xmaxs = np.minimum(anchors_xmax, frame_window_width)
    xmaxs = xmaxs + np.expand_dims(start_frames, axis=1)
    # xmaxs = xmaxs / feat_tem_width * frame_window_width + np.expand_dims(start_frames, axis=1)

    # expand video_name
    vid_name_df = list()
    num_tem_loc = anchors_xmin.shape[1]
    for i in range(len(video_names)):
        vid_names = [video_names[i]] * num_tem_loc
        vid_name_df.extend(vid_names)
    out_df['video_name'] = vid_name_df

    # reshape numpy array
    # Notice: this is not flexible enough
    num_element = anchors_xmin.shape[0] * anchors_xmin.shape[1]
    xmins_tmp = np.reshape(xmins, num_element)
    out_df['xmin'] = xmins_tmp
    xmaxs_tmp = np.reshape(xmaxs, num_element)
    out_df['xmax'] = xmaxs_tmp

    # scores_action = cls_scores[:, :, 1:]
    scores_action = cls_scores
    max_values = np.amax(scores_action, axis=2)
    conf_tmp = np.reshape(max_values, num_element)
    out_df['conf'] = conf_tmp
    max_idxs = np.argmax(scores_action, axis=2)
    max_idxs = max_idxs + 1
    cate_idx_tmp = np.reshape(max_idxs, num_element)
    # Notice: convert index into category type
    class_real = cfg.DATASET.CLASS_IDX
    for i in range(len(cate_idx_tmp)):
        cate_idx_tmp[i] = class_real[int(cate_idx_tmp[i])]
    out_df['cate_idx'] = cate_idx_tmp

    out_df = out_df[cfg.TEST.OUTDF_COLUMNS_AB]
    return out_df
