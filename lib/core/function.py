import torch
import torch.nn as nn
import pandas as pd
import os

from core.utils_af import result_process_af


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


def train(cfg, train_loader, model, optimizer,epoch):
    model.train()
    loss_record = 0
    cls_loss_record,iou_loss_record,dir_loss_record = 0,0,0
    for feat_spa,feat_tem, boxes, label, action_num in train_loader:
        optimizer.zero_grad()
        feature = torch.cat((feat_spa, feat_tem), dim=1)        #[batch,2048,512]
        feature = feature.type_as(dtype)
        boxes = boxes.float().type_as(dtype)
        label = label.type_as(dtypel)

        label = label[..., None]
        targets = torch.cat([label, boxes], dim=-1)
        extra_info = {
            'epoch': epoch
        }


        loss = model(feature,targets,extra_info)

        loss['total_loss'].backward()
        optimizer.step()
        loss_record = loss_record + loss['total_loss'].item()
        cls_loss_record += loss['loss_cls'].item()
        iou_loss_record += loss['loss_iou'].item()
        dir_loss_record += loss['loss_dfl'].item()

    return loss_record / len(train_loader), cls_loss_record / len(train_loader), \
           iou_loss_record / len(train_loader), dir_loss_record / len(train_loader)


def evaluation(val_loader, model, epoch, cfg):
    model.eval()
    out_df_af = pd.DataFrame(columns=cfg.TEST.OUTDF_COLUMNS_AF)
    for feat_spa,feat_tem, begin_frame, video_name in val_loader:
        begin_frame = begin_frame.detach().numpy()

        feature = torch.cat((feat_spa, feat_tem), dim=1)
        feature = feature.type_as(dtype)
        out_preds = model(feature)


        preds_cls, preds_reg = out_preds[...,3:],out_preds[...,:3]
        preds_cls = preds_cls.detach().cpu().numpy()

        xmins = preds_reg[:, :, 0]
        xmins = xmins.detach().cpu().numpy()
        xmaxs = preds_reg[:, :, 1]
        xmaxs = xmaxs.detach().cpu().numpy()

        tmp_df_af = result_process_af(video_name, begin_frame, preds_cls, xmins, xmaxs, cfg)
        out_df_af = pd.concat([out_df_af, tmp_df_af], sort=True)


    return  out_df_af
