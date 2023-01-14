import os
import glob
import pandas as pd
import numpy as np
import argparse
import math
from torch.nn.functional import threshold



def all_score(TP1, TP2, N1, N2, recall1, recall2, recall_all):
    if TP1 == 0 and TP2 != 0:
        precision1 = 0
        precision2 = 1.0 * TP2 / N2
        precision_all = 1.0 * (TP1 + TP2) / (N1 + N2)
        F1_SCORE_M1 = 0
        F1_SCORE_M2 = 2 * (recall2 * precision2) / (recall2 + precision2)
        F1_SCORE = 2 * (recall_all * precision_all) / (recall_all + precision_all)
    elif TP1 != 0 and TP2 == 0:
        precision1 = 1.0 * TP1 / N1
        precision2 = 0
        precision_all = 1.0 * (TP1 + TP2) / (N1 + N2)
        F1_SCORE_M1 = 2 * (recall1 * precision1) / (recall1 + precision1)
        F1_SCORE_M2 = 0
        F1_SCORE = 2 * (recall_all * precision_all) / (recall_all + precision_all)
    elif TP1 == 0 and TP2 == 0:
        precision1 = 0
        precision2 = 0
        precision_all = 0
        F1_SCORE_M1 = 0
        F1_SCORE_M2 = 0
        F1_SCORE = 0
    else:
        precision1 = 1.0 * TP1 / N1
        precision2 = 1.0 * TP2 / N2
        precision_all = 1.0 * (TP1 + TP2) / (N1 + N2)
        F1_SCORE_M1 = 2 * (recall1 * precision1) / (recall1 + precision1)
        F1_SCORE_M2 = 2 * (recall2 * precision2) / (recall2 + precision2)
        F1_SCORE = 2 * (recall_all * precision_all) / (recall_all + precision_all)

    return F1_SCORE_M1, F1_SCORE_M2, F1_SCORE, precision_all


def main_threshold(path, dataset, annotation, version, label_frequency, start_threshold, max_num_pos):
    files_tmp = os.listdir(path)
    files = sorted(files_tmp, key=lambda x: int(x[-2:]))
    ann_csv = pd.read_csv(annotation)
    test_path_temp = [os.path.join(path, i, 'test_detection') for i in files]
    txts = glob.glob(os.path.join(test_path_temp[0], '*.txt'))

    txts = [int(i.split('_')[-1].split('.')[0]) for i in txts]
    txts.sort()

    best, best_m1, best_m2 = 0, 0, 0
    best_recall = 0

    for e in range(1, 32):
        tmp_f1_cores, TP_tmp, FP_tmp, FN_tmp, p_tmp, r_tmp, k_tmp = 0, 0, 0, 0, 0, 0, 0
        txt_index = txts[e]
        # all subjects in the same epoch
        test_path = [os.path.join(i, 'test_' + str(txt_index).zfill(2) + '.txt') for i in
                     test_path_temp]
        # confirm the best threshold
        for k_temp in range(start_threshold, 700, 1):
            k = 1.0 * k_temp / 1000
            FP, FN, TP = 0, 0, 0
            TP1, TP2 = 0, 0
            N1, N2, N_all = 0, 0, 0
            length_count = list()
            write_list = list()
            length_pre = list()
            # every subject in one file (200x)
            for i in test_path:
                with open(i, 'r') as f:
                    all_lines = f.readlines()
                all_lines = [h.split('\t') for h in all_lines]

                # divide all gts of every video
                tmp_video = all_lines[0][0]
                count = 1
                tmp_list = list()
                all_test = dict()
                all_video = list(set([name[0] for name in all_lines]))
                for tv in all_video:
                    tmp_video = tv
                    for j in range(len(all_lines)):
                        if all_lines[j][0] == tmp_video:
                            tmp_list.append(all_lines[j])
                    all_test[count] = tmp_list
                    count = count + 1
                    tmp_list = list()
                # number of GT of every video
                num_of_video = len(all_test.keys())

                # least len of GT
                part_tmp = list()
                # select predictions of every video (prob > threshold)
                for i in range(num_of_video):
                    tmp_one_video = list(all_test.values())[i]
                    part = [o for o in tmp_one_video if float(o[-1][:-2]) > k]
                    # N1: number of precictions of macro-expressions
                    # N2: number of precictions of micro-expressions
                    # N_all: number of precictions
                    if len(part) > max_num_pos:
                        part = part[:max_num_pos]
                    N_all = N_all + len(part)
                    N1 = N1 + len([o for o in part if o[-2] == '1'])
                    N2 = N2 + len([o for o in part if o[-2] == '2'])

                    if not part:
                        continue
                    part_tmp.append(part)
                part_pre = part_tmp

                # predictions: sorted by prob
                part_pre = [sorted(i, key=lambda x: int(float(x[1]))) for i in part_pre]

                # calculate iou between every prediction with GT
                for video_num, pre in enumerate(part_pre):
                    video_name_list = list(set(ann_csv.video.values[:].tolist()))
                    video_name_list.sort()

                    # identify the current video
                    video_name_last = part_pre[video_num][0][0]
                    if dataset == 'cas(me)^2':
                        video_name_part = 's' + video_name_last[:2]
                        video_name = os.path.join(video_name_list[0].split('/s')[0], video_name_part, video_name_last)
                    else:
                        video_name = os.path.join(video_name_list[0][:-4], str(video_name_last).zfill(3))

                    # select startframes of current video
                    video_name = video_name.replace("\\", '/')
                    video_ann_df = ann_csv[ann_csv.video == video_name]
                    act_start_video = video_ann_df['startFrame'].values[:]
                    # select indexes of startframes of current video
                    indexes = np.argsort(act_start_video)
                    # labels and endframes are sorted by indexes from actual start frames
                    act_end_video = video_ann_df['endFrame'].values[:]
                    act_end_video = np.array(act_end_video)[indexes]
                    labels = video_ann_df['type_idx'].values[:]
                    labels = np.array(labels)[indexes]
                    act_start_video.sort()

                    pre = np.array(pre)
                    pre_start = pre[:, 1].astype(float).astype(np.int64) * int(label_frequency)
                    pre_end = pre[:, 2].astype(float).astype(np.int64) * int(label_frequency)
                    pre_label = pre[:, 3].astype(float).astype(np.int64)

                    start_tmp = list()
                    end_tmp = list()
                    match_tmp_index = list()
                    for m in range(len(act_start_video)):
                        video_label = video_name_last[:7]
                        act_start = int(act_start_video[m])
                        act_end = int(act_end_video[m])
                        iou = (np.minimum(pre_end, act_end) - np.maximum(pre_start, act_start)) / (
                                np.maximum(pre_end, act_end) - np.minimum(pre_start, act_start))
                        max_iou = np.max(iou)
                        max_index = np.argmax(iou)
                        if max_iou >= 0.5 and labels[m] == int(float(pre[max_index][-2])):
                            tmp_write_list = [video_label, pre_start[max_index], pre_end[max_index], act_start, act_end,
                                              max_iou, pre_label[max_index], labels[m], 'TP']
                            write_list.append(tmp_write_list)
                            if labels[m] == 1:
                                TP1 = TP1 + 1
                            elif labels[m] == 2:
                                TP2 = TP2 + 1
                            start_tmp.append(pre_start[max_index])
                            end_tmp.append(pre_end[max_index])
                            match_tmp_index.append(m)
                        else:
                            tmp_write_list = [video_label, pre_start[max_index], pre_end[max_index], act_start, act_end,
                                              max_iou, pre_label[max_index], labels[m], 'FP']
                            write_list.append(tmp_write_list)

                    # pre_remain_e = [i for i in pre_end_remain if i not in end_tmp]
                    no_match_index = [i for i in range(len(act_start_video)) if i not in match_tmp_index]
                    # write_remain = [[video_label, i, pre_end[pre_start==i][0], '_', '_', 'FN'] for i in pre_remain_s]
                    write_remain = [
                        [video_label, '_', '_', act_start_video[i], act_end_video[i], 0, labels[i], labels[i], 'FN'] for
                        i in no_match_index]
                    write_list = write_list + write_remain

            # calculate F1_score
            # M_all need to calculate in SAMM
            # M1： Number of macro-expressions
            # M2： Number of micro-expressions
            if dataset == 'cas(me)^2' or dataset == 'cas(me)^2_merge':
                M1 = 282
                M2 = 84
            else:
                M1 = 340
                M2 = 159
            recall1 = 1.0 * TP1 / M1
            recall2 = 1.0 * TP2 / M2
            recall_all = 1.0 * (TP1 + TP2) / (M1 + M2)
            if recall_all > best_recall:
                best_recall = recall_all
                print('best', recall_all)

            F1_SCORE_M1, F1_SCORE_M2, F1_SCORE, precision_all = all_score(TP1, TP2, N1, N2, recall1, recall2,
                                                                          recall_all)
            # Micro-expression
            if F1_SCORE_M1 > best_m1:
                best_m1 = F1_SCORE_M1
                compute_ME(TP1, N1, 1, M1)
                precision_1 = TP1 / N1
                print("Best MaE : recall:" + str(recall1) + " precision: " + str(precision_1) + " f1_score: " + str(
                    best_m1) + "  threshold:" + str(k))

            if F1_SCORE_M2 > best_m2:
                best_m2 = F1_SCORE_M2
                compute_ME(TP2, N2, 2,M2)
                precision_2 = TP2 / N2
                print("Best MiE : recall:" + str(recall2) + " precision: " + str(precision_2) + " f1_score: " + str(
                    best_m2) + "  threshold:" + str(k))



            if F1_SCORE_M2 > tmp_f1_cores:
                tmp_f1_cores = F1_SCORE_M2
                TP_tmp, FP_tmp, FN_tmp = TP2,N2-TP2,M2-TP2
                r_tmp = TP2 / M2
                p_tmp = TP2 / N2
                k_tmp = k


        print("------------------------------------------------------")
        print("epoch:", e + 1)
        print("Best MiE: " + "TP: " + str(TP_tmp) + "FP: " + str(FP_tmp) + "FN: " + str(FN_tmp))
        print("Best MiE : recall:" + str(r_tmp) + "precision: " + str(p_tmp) + "f1_score: " + str(
            tmp_f1_cores) + "  threshold:" + str(k_tmp))
        print("------------------------------------------------------")



def compute_ME(TP2, N2, way,sample_num):
    TP = TP2
    FP = N2 - TP
    FN = sample_num - TP
    if way == 2:
        print("Best Micro: " + "TP: " + str(TP) + " FP: " + str(FP) + " FN: " + str(FN))
    elif way == 1:
        print("Best Macro: " + "TP: " + str(TP) + " FP: " + str(FP) + " FN: " + str(FN))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')

    # ori
    parser.add_argument('--path', type=str, default=r'D:\gxp\code\3D_work\me_spotting\output_V28\samm_5_5_3')
    parser.add_argument('--ann', type=str, default=r'D:\Programming\SharedCodes\MULT-MicroExpressionSpot\data\samm_annotation_v2.csv')
    # parser.add_argument('--path', type=str, default='D:\Programming\MES-LSSNet\mer_spot-master\output_V28\cas(me)^2')
    # parser.add_argument('--ann', type=str, default=r'D:\Programming\MES-LSSNet\mer_spot-master\casme2_annotation.csv')
    # parser.add_argument('--dataset', type=str, default='cas(me)^2')
    parser.add_argument('--dataset', type=str, default='samm')
    parser.add_argument('--version', type=int, default=28)
    parser.add_argument('--top_k', type=bool, default=False)
    parser.add_argument('--label_frequency', type=float, default=5.0)
    parser.add_argument('--start_threshold', type=int, default=70)
    parser.add_argument('--most_pos_num', type=int, default=70)

    args = parser.parse_args()

    path = args.path
    dataset = args.dataset
    ann = args.ann
    version = args.version
    label_frequency = args.label_frequency
    start_threshold = args.start_threshold
    max_num_pos = args.most_pos_num

    main_threshold(path, dataset, ann, version, label_frequency, start_threshold, max_num_pos)
