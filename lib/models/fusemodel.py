import math
import torch.nn as nn
import torch
import torch.nn.functional as F

from lib.core.atss_assigner import ATSSAssigner, TaskAlignedAssigner
from lib.core.loss import VarifocalLoss, FocalLoss, BboxLoss
from lib.core.utils_af import batch_distance2bbox


def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output

class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)


class LocalMaskedMHCA(nn.Module):
    def __init__(
            self,
            n_embd,  # dimension of the output features
            n_head,  # number of heads in multi-head self-attention
            window_size=8,  # size of the local attention window
            n_qx_stride=1,  # dowsampling stride for query and input
            n_kv_stride=1,  # downsampling stride for key and value
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for projection op
            use_rel_pe=False  # use relative position encoding
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap = window_size // 2
        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2

        self.query_conv = nn.Conv1d(
            self.n_embd, self.n_embd, kernel_size, stride, padding=padding, groups=self.n_embd, bias=False
        )

        # layernorm
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2

        self.key_conv = nn.Conv1d(
            self.n_embd, self.n_embd, kernel_size, stride, padding=padding, groups=self.n_embd, bias=False
        )

        self.key_norm = LayerNorm(self.n_embd)

        self.value_conv = nn.Conv1d(
            self.n_embd, self.n_embd, kernel_size, stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.value_norm = LayerNorm(self.n_embd)


        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)



    @staticmethod
    def _chunk(x, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # x: B x nh, T, hs
        # non-overlapping chunks of size = 2w -> B x nh, T//2w, 2w, hs

        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2

        # B x nh, #chunks = T//w - 1, 2w, hs
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        # `== 1` converts to bool or uint8
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        # `== 1` converts to bool or uint8
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1).
        x = nn.functional.pad(
            x, (0, window_overlap + 1)
        )
        # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        x = x.view(total_num_heads, num_chunks, -1)
        # total_num_heads x num_chunks x window_overlap*window_overlap
        x = x[:, :, :-window_overlap]
        x = x.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(
            self, query, key, num_heads, window_overlap
    ):

        # query / key: B*nh, T, hs
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # B * num_heads, head_dim, #chunks=(T//w - 1), 2w
        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)


        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (chunk_query, chunk_key))

        # convert diagonals into columns
        # B * num_heads, #chunks, 2w, 2w+1
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )


        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )  # [8,256,9,19]


        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                                :, :, :window_overlap, : window_overlap + 1
                                                                ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                               :, -1, window_overlap:, : window_overlap + 1
                                                               ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
                                                               :, :, -(window_overlap + 1): -1, window_overlap + 1:
                                                               ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
                                                                              :, 0, : window_overlap - 1,
                                                                              1 - window_overlap:
                                                                              ]


        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
            self, attn_probs, value, num_heads, window_overlap
    ):

        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x):
        B, C, T = x.size()

        # step 1: depth convolutions
        # query conv -> (B, nh * hs, T')
        q = self.query_conv(x)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k = self.key_conv(x)
        k = self.key_norm(k)
        v = self.value_conv(x)
        v = self.value_norm(v)

        # step 2: query, key, value transforms & reshape
        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        # (B, nh * hs, T) -> (B, nh, T, hs)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # view as (B * nh, T, hs)
        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        # step 3: compute local self-attention with rel pe and masking
        q *= self.scale
        # chunked query key attention -> B, T, nh, 2w+1 = window_size
        att = self._sliding_chunks_query_key_matmul(
            q, k, self.n_head, self.window_overlap)


        # ignore input masking for now
        att = nn.functional.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # step 4: compute attention value product + output projection
        # chunked attn value product -> B, nh, T, hs
        out = self._sliding_chunks_matmul_attn_probs_value(
            att, v, self.n_head, self.window_overlap)
        # transpose to B, nh, hs, T -> B, nh*hs, T
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        out = self.proj_drop(self.proj(out))
        return out


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
            self,
            num_channels,
            eps=1e-5,
            affine=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.tensor(1, dtype=torch.float32),
                requires_grad=True
            )
            self.bias = nn.Parameter(
                torch.tensor(0, dtype=torch.float32),
                requires_grad=True
            )
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


# basic block
class Base_ConvBnSilu(nn.Module):
    def __init__(self, in_dim, out_dim, stride, kernel):
        super(Base_ConvBnSilu, self).__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel, stride=stride, padding=pad)
        # self.bn = nn.BatchNorm1d(out_dim)
        self.bn = LayerNorm(out_dim)

    def forward(self, x):
        return torch.nn.functional.silu(self.bn(self.conv(x)))


# block in C3
class BottleNeck(nn.Module):
    def __init__(self, in_dim, residual=False):
        super(BottleNeck, self).__init__()
        self.residual = residual
        self.b1 = Base_ConvBnSilu(in_dim, in_dim, 1, 1)
        self.b2 = Base_ConvBnSilu(in_dim, in_dim, 1, 3)

    def forward(self, x):
        if self.residual: return x + self.b2(self.b1(x))
        return self.b2(self.b1(x))



class C3_Block(nn.Module):
    def __init__(self, in_dim, rep_num, out_channel, bottleNeck1=True):
        super(C3_Block, self).__init__()
        self.branch1 = Base_ConvBnSilu(in_dim, int(in_dim / 2), 1, 1)
        self.BottleNecks = nn.ModuleList()
        for _ in range(rep_num):
            if _ == 0:
                self.BottleNecks.append(nn.Sequential(
                    Base_ConvBnSilu(in_dim, int(in_dim / 2), 1, 1),
                    BottleNeck(int(in_dim / 2), bottleNeck1)
                ))
            else:
                self.BottleNecks.append(nn.Sequential(
                    Base_ConvBnSilu(int(in_dim / 2), int(in_dim / 2), 1, 1),
                    BottleNeck(int(in_dim / 2), bottleNeck1)
                ))
        self.fuse = Base_ConvBnSilu(in_dim, out_channel, 1, 1)

    def forward(self, x):
        branch1 = self.branch1(x)
        for neck in self.BottleNecks:
            x = neck(x)
        return self.fuse(torch.concat([branch1, x], dim=1))



class SPPF(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.b1 = Base_ConvBnSilu(in_dim, int(in_dim / 2), 1, 1)
        self.maxpool = nn.MaxPool1d(5, 1, padding=2)
        self.b2 = Base_ConvBnSilu(int(in_dim * 2), in_dim, 1, 1)

    def forward(self, x):
        x = self.b1(x)
        o1 = self.maxpool(x)
        o2 = self.maxpool(o1)
        o3 = self.maxpool(o2)
        return self.b2(torch.cat([x, o1, o2, o3], dim=1))


############### backbone ##############
class BaseFeatureNet(nn.Module):
    def __init__(self, cfg):
        super(BaseFeatureNet, self).__init__()

        self.b1 = Base_ConvBnSilu(cfg.MODEL.IN_FEAT_DIM, int(cfg.MODEL.IN_FEAT_DIM / 2), 1, 3)
        self.c3 = C3_Block(int(cfg.MODEL.IN_FEAT_DIM / 2), 1, int(cfg.MODEL.IN_FEAT_DIM / 4))

    def forward(self, x):
        return self.c3(self.b1(x))


############### FPN ##############
class FPNNeck(nn.Module):
    def __init__(self, cfg):
        super(FPNNeck, self).__init__()
        self.cfg = cfg
        self.base_feature_net = BaseFeatureNet(cfg)
        self.fpns = nn.ModuleList()
        self.downs = nn.ModuleList()  # downsample layer after fpn
        for layer in range(cfg.MODEL.NUM_LAYERS-1):
            in_channel = cfg.MODEL.BASE_FEAT_DIM if layer == 0 else cfg.MODEL.LAYER_DIMS[layer - 1]
            out_channel = cfg.MODEL.LAYER_DIMS[layer]
            self.fpns.append(C3_Block(in_channel, 1, in_channel, bottleNeck1=True))
            self.downs.append(Base_ConvBnSilu(in_channel, out_channel, 2, 3))
        self.sppf = SPPF(cfg.MODEL.LAYER_DIMS[-1])

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="linear",align_corners=True)

        # fusion model
        self.fuse = nn.ModuleList()
        for _ in range(cfg.MODEL.NUM_LAYERS-1):
            self.fuse.append(Base_ConvBnSilu(cfg.MODEL.LAYER_DIMS[_]*2,cfg.MODEL.LAYER_DIMS[_],1,3))


    def forward(self, x):
        results = []
        feat = self.base_feature_net(x)
        idx = 0
        for layer, down_sample in zip(self.fpns, self.downs):
            feat = layer(feat)
            results.append(feat)
            feat = down_sample(feat)
            if idx == len(self.fpns) - 1:
                feat = self.sppf(feat)
                results.append(feat)
            idx += 1
        # upsample
        for i in range(self.cfg.MODEL.NUM_LAYERS-1,0,-1):
            tmp_fuse = torch.cat((results[i-1],self.upsample(results[i])),dim=1)
            fuse_feature = self.fuse[self.cfg.MODEL.NUM_LAYERS-1-i](tmp_fuse)
            results[i-1] = fuse_feature
        return tuple(results)


############### Neck ######################
class NeckTransformer(nn.Module):
    def __init__(self, cfg):
        super(NeckTransformer, self).__init__()
        self.num_class = cfg.DATASET.NUM_CLASSES
        self.rep_num = cfg.MODEL.NUM_LAYERS
        self.necks = nn.ModuleList()
        for _ in range(self.rep_num):
            self.necks.append(LocalMaskedMHCA(cfg.MODEL.HEAD_DIM, 4))

    def forward(self, feats):
        out = []
        for idx, feat in enumerate(feats):
            out.append(self.necks[idx](feat))
        out = out[::-1]
        return out

################### Head ###################
class PredHead(nn.Module):

    def __init__(self, cfg):
        super(PredHead, self).__init__()
        # dist_max in cls model
        self.dist_max = 16
        # Attentions of weight sharing
        self.stem_cls = ESEAttn(cfg.MODEL.REDU_CHA_DIM)
        self.stem_reg = ESEAttn(cfg.MODEL.REDU_CHA_DIM)


        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for _ in range(cfg.MODEL.NUM_LAYERS):
            self.pred_cls.append(nn.Conv1d(cfg.MODEL.REDU_CHA_DIM,cfg.DATASET.NUM_CLASSES,kernel_size=3,padding=1))
            self.pred_reg.append(nn.Conv1d(cfg.MODEL.REDU_CHA_DIM,(self.dist_max +1 )*2,kernel_size=3,padding=1))

        # init  "pred_cls and pred_reg"
        self._init_weights()

    def forward(self, feats,train=False):
        cls_score_list, reg_distri_list = [], []
        if train:
            for i, feat in enumerate(feats):
                avg_feat = F.adaptive_avg_pool1d(feat, (1))
                cls_logit = self.pred_cls[i](self.stem_cls(feat, avg_feat) +
                                             feat)
                reg_distri = self.pred_reg[i](self.stem_reg(feat, avg_feat))
                # cls and reg
                cls_score = F.sigmoid(cls_logit)
                cls_score_list.append(cls_score.permute((0, 2, 1)))
                reg_distri_list.append(reg_distri.permute((0, 2, 1)))
            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)

            return cls_score_list,reg_distri_list
        else:
            for i, feat in enumerate(feats):
                avg_feat = F.adaptive_avg_pool1d(feat, (1))
                cls_logit = self.pred_cls[i](self.stem_cls(feat, avg_feat) +
                                             feat)
                reg_distri = self.pred_reg[i](self.stem_reg(feat, avg_feat))
                # cls and reg
                cls_score_list.append(cls_logit)
                reg_distri_list.append(reg_distri)
            return cls_score_list,reg_distri_list

    def _init_weights(self, prior_prob=0.01):
        for conv in self.pred_cls:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.pred_reg:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

class ESEAttn(nn.Module):
    def __init__(self, feat_channels):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv1d(feat_channels, feat_channels, 1)
        self.sig = nn.Sigmoid()
        self.conv = Base_ConvBnSilu(feat_channels, feat_channels, 1,1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc.weight, mean=0, std=0.001)

    def forward(self, feat, avg_feat):
        weight = self.sig(self.fc(avg_feat))
        return self.conv(feat * weight)

class FuseModel(nn.Module):
    def __init__(self, cfg):
        super(FuseModel, self).__init__()
        # backbone
        self.features = FPNNeck(cfg)
        # neck
        self.necks = NeckTransformer(cfg)
        # head
        self.head = PredHead(cfg)
        # #Param
        self.temp_length = cfg.MODEL.TEMPORAL_LENGTH
        self.strides =  cfg.MODEL.TEMPORAL_STRIDE
        self.reg_max = 16
        self.static_assigner_epoch = 10
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.use_varifocal_loss = True
        self.loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 }


        # projection conv
        self.proj_conv = nn.Conv1d(self.reg_max + 1, 1, 1, bias=False)
        self._init_weights()

        # atss_assigner
        self.atss_assign = ATSSAssigner(num_classes=cfg.DATASET.NUM_CLASSES)
        self.assigner = TaskAlignedAssigner(topk=9,num_classes=cfg.DATASET.NUM_CLASSES)
        self.varifocal_loss = VarifocalLoss().cuda()
        self.focal_loss = FocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max).cuda()


    def _init_weights(self, prior_prob=0.01):
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = torch.nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)


    def forward(self,x,targets=None,extra_info=None):
        if targets!=None:
            return self.forward_train(x,targets,extra_info)
        else:
            return self.forward_eval(x)

    def forward_eval(self,x):
        # backbone
        features = self.features(x)
        # feat
        feats = self.necks(features)
        anchor_points, stride_tensor = self._generate_anchors(self.temp_length[::-1],self.strides[::-1])
        cls_score_list, reg_dist_list = [], []
        preds_cls,preds_reg = self.head(feats,False)
        for i in range(len(preds_reg)):
            b, _, tmp_length = preds_cls[i].shape
            reg_dist = preds_reg[i]
            cls_logit = preds_cls[i]
            reg_dist = reg_dist.reshape([-1, 2, self.reg_max + 1, tmp_length]).permute(
                0, 2, 1, 3)
            reg_dist = self.proj_conv(F.softmax(reg_dist, dim=1))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, self.num_classes, tmp_length]))
            # reg_dist_list.append(reg_dist.reshape([b, 2, tmp_length]))
            reg_dist_list.append(reg_dist.squeeze(1))

        cls_score_list = torch.cat(cls_score_list, axis=-1)
        reg_dist_list = torch.cat(reg_dist_list, axis=-1)


        pred_bboxes = batch_distance2bbox(anchor_points.unsqueeze(-1),
                                       reg_dist_list.permute(0,2,1))
        pred_bboxes *= stride_tensor
        return torch.cat(
            [
                pred_bboxes,    #
                torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                cls_score_list.permute(0, 2, 1)
            ],
            axis=-1)

    def forward_train(self, x,targets,extra_info):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            self.generate_anchors_for_grid_cell(self.temp_length,self.strides, device=x.device)

        features = self.features(x)
        feats= self.necks(features)
        cls_score_list, reg_distri_list = self.head(feats,True)


        return self.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets, extra_info)

    def get_loss(self, head_outs, targets, extra_info):
        pred_scores, pred_distri, anchors, \
        anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor

        pred_bboxes = self.bbox_decode(anchor_points_s,
                                       pred_distri)
        gt_labels = targets[:, :, :1]

        gt_bboxes = targets[:, :, 1:]
        pad_gt_mask = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        if extra_info['epoch'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = self.atss_assign(
                anchors,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes,
                pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            alpha_l = -1

        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels, self.num_classes + 1)[..., :-1]
            loss_cls = self.varifocal_loss(pred_scores, assigned_scores,
                                           one_hot_label)
        else:
            loss_cls = self.focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        # loss_cls /= assigned_scores_sum
        loss_cls /= (assigned_scores_sum + 1e-6)

        loss_l1, loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                                                     assigned_labels, assigned_bboxes, assigned_scores,
                                                     assigned_scores_sum)

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        out_dict = {
            'total_loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
        }
        return out_dict

    def generate_anchors_for_grid_cell(self,temp_length, fpn_strides, grid_cell_size=8.0, grid_cell_offset=0.5,
                                       device='cpu'):
        r"""
        Like ATSS, generate anchors based on grid size.
        """
        assert len(temp_length) == len(fpn_strides)
        anchors = []
        anchor_points = []
        num_anchors_list = []
        stride_tensor = []
        fpn_strides = fpn_strides[::-1]
        temp_length = temp_length[::-1]
        for length, stride in zip(temp_length, fpn_strides):
            cell_half_size = grid_cell_size * stride * 0.5
            shift_t = (torch.arange(end=length, device=device) + grid_cell_offset) * stride
            anchor = torch.stack(
                [
                    shift_t - cell_half_size, shift_t + cell_half_size
                ],
                axis=-1).clone().to(torch.float32)
            anchor_point = torch.stack(
                [shift_t], axis=-1).clone().to(torch.float32)

            anchors.append(anchor.reshape([-1, 2]))
            anchor_points.append(anchor_point.reshape([-1, 1]))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(
                torch.full(
                    [num_anchors_list[-1], 1], stride, dtype=torch.float32))
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).cuda()
        stride_tensor = torch.cat(stride_tensor).cuda()
        return anchors, anchor_points, num_anchors_list, stride_tensor


    def bbox_decode(self, anchor_points, pred_dist):
        batch_size, n_anchors, _ = pred_dist.shape
        pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 2, self.reg_max + 1), dim=-1).matmul(self.proj)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _generate_anchors(self, tmp_lengths,strides, device='cuda:0'):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i in range(len(tmp_lengths)):
            tmp_length = tmp_lengths[i]
            stride = strides[i]
            shift_t = torch.arange(end=tmp_length, device=device) + 0.5  # grid_offset
            anchor_points.append(shift_t)
            stride_tensor.append(
                torch.full(
                    (tmp_length, 1), stride, dtype=torch.float, device=device))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor


if __name__ == '__main__':
    import sys

    sys.path.append(r'D:\Programming\SharedCodes\MULT-MicroExpressionSpot\lib')
    from config import cfg, update_config

    cfg_file = r'D:\Programming\SharedCodes\MULT-MicroExpressionSpot\data\SAMM_5.yaml'
    update_config(cfg_file)

    net = FuseModel(cfg).cuda()
    total = sum([param.nelement() for param in net.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))


