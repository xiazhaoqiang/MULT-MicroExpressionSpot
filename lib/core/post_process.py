import pandas as pd
import os
import numpy as np




def temporal_nms(df, cfg):
    '''
    temporal nms
    I should understand this process
    '''

    type_set = list(set(df.cate_idx.values[:]))
    # type_set.sort()

    # returned values
    rstart = list()
    rend = list()
    rscore = list()
    rlabel = list()

    for t in type_set:
        label = t
        tmp_df = df[df.cate_idx == t]

        start_time = np.array(tmp_df.xmin.values[:])
        end_time = np.array(tmp_df.xmax.values[:])
        scores = np.array(tmp_df.conf.values[:])

        duration = end_time - start_time
        order = scores.argsort()[::-1]

        keep = list()
        while (order.size > 0) and (len(keep) < cfg.TEST.TOP_K_RPOPOSAL):
            i = order[0]
            keep.append(i)
            tt1 = np.maximum(start_time[i], start_time[order[1:]])
            tt2 = np.minimum(end_time[i], end_time[order[1:]])
            intersection = tt2 - tt1
            union = (duration[i] + duration[order[1:]] - intersection).astype(float)
            # iou = intersection / union
            iou = intersection / (union + 1e-6)

            inds = np.where(iou <= cfg.TEST.NMS_TH)[0]
            order = order[inds + 1]

        # record the result
        for idx in keep:
            rlabel.append(label)
            rstart.append(float(start_time[idx]))
            rend.append(float(end_time[idx]))
            rscore.append(scores[idx])

    new_df = pd.DataFrame()
    new_df['start'] = rstart
    new_df['end'] = rend
    new_df['score'] = rscore
    new_df['label'] = rlabel
    return new_df



def final_result_process(out_df, epoch,subject, cfg):
    path_tmp = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject, cfg.TEST.PREDICT_TXT_FILE)
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)

    res_txt_file = os.path.join(path_tmp, 'test_' + str(epoch).zfill(2)+'.txt')
    if os.path.exists(res_txt_file):
        os.remove(res_txt_file)
    
    f = open(res_txt_file, 'a')


    video_name_list = list(set(out_df.video_name.values[:]))

    for video_name in video_name_list:
        tmpdf = out_df[out_df.video_name == video_name]

        type_set = list(set(tmpdf.cate_idx.values[:]))
        df_nms = temporal_nms(tmpdf, cfg)
        # ensure there are most 200 proposals
        df_vid = df_nms.sort_values(by='score', ascending=False)

        for i in range(min(len(df_vid), cfg.TEST.TOP_K_RPOPOSAL)):
            start_time = df_vid.start.values[i]
            end_time = df_vid.end.values[i]
            label = df_vid.label.values[i]
            strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (video_name, float(start_time), float(end_time), label, df_vid.score.values[i])
            f.write(strout)
    f.close()
