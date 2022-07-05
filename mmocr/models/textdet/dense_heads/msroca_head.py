import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
import torch.nn.functional as F

from mmocr.models.textdet.postprocess import decode
from ..postprocess.wrapper import poly_nms
from .head_mixin import HeadMixin
from mmocr.models.textdet.modules import RCAB, ROAM, CrossFormer


@HEADS.register_module()
class MSROCAHead(HeadMixin, nn.Module):
    """
    Args:
        in_channels (int): The number of input channels.
        scales (list[int]) : The scale of each layer.
        fourier_degree (int) : The maximum Fourier transform degree k.
        num_sample (int) : The sampling points number of regression
            loss. If it is too small, FCEnet tends to be overfitting.
        score_thr (float) : The threshold to filter out the final
            candidates.
        nms_thr (float) : The threshold of nms.
        alpha (float) : The parameter to calculate final scores. Score_{final}
            = (Score_{text region} ^ alpha)
            * (Score{text center region} ^ beta)
        beta (float) :The parameter to calculate final scores.
    """

    def __init__(self,
                 in_channels,
                 scales,
                 fourier_degree=5,
                 num_sample=50,
                 num_reconstr_points=50,
                 decoding_type='fcenet',
                 loss=dict(type='MSROCALoss'),
                 score_thr=0.3,
                 nms_thr=0.1,
                 alpha=1.0,
                 beta=1.0,
                 text_repr_type='poly',
                 train_cfg=None,
                 test_cfg=None):

        super().__init__()
        assert isinstance(in_channels, int)

        self.downsample_ratio = 1.0
        self.in_channels = in_channels
        self.scales = scales
        self.fourier_degree = fourier_degree
        self.sample_num = num_sample
        self.num_reconstr_points = num_reconstr_points
        loss['fourier_degree'] = fourier_degree
        loss['num_sample'] = num_sample
        self.decoding_type = decoding_type
        self.loss_module = build_loss(loss)
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.alpha = alpha
        self.beta = beta
        self.text_repr_type = text_repr_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.out_channels_cls = 4
        self.out_channels_reg = (2 * self.fourier_degree + 1) * 2

        self.rcab = RCAB(n_feat=self.in_channels, kernel_size=3, reduction=16)
        self.loa = ROAM(in_channels=2)
        self.crossformer = CrossFormer(img_size=100, in_chans=256, num_classes=2)

        self.out_conv_cls = nn.Conv2d(
            self.in_channels,
            self.out_channels_cls,
            kernel_size=3,
            stride=1,
            padding=1)
        self.out_conv_reg = nn.Conv2d(
            self.in_channels,
            self.out_channels_reg,
            kernel_size=3,
            stride=1,
            padding=1)
        self.out_conv = nn.Conv2d(
            768,
            256,
            kernel_size=1,
            stride=1,
            padding=0)
        self.context_conv = nn.Conv2d(
            96,
            256,
            kernel_size=1,
            stride=1,
            padding=0)

        self.init_weights()

    def init_weights(self):
        normal_init(self.out_conv_cls, mean=0, std=0.01)
        normal_init(self.out_conv_reg, mean=0, std=0.01)

    def context_module(self, feats):
        outs = []
        for i in range(len(feats)):
            outs.append(F.interpolate(feats[i], size=feats[0].shape[2:], mode='nearest'))
        out = torch.cat(outs, dim=1)
        out = self.out_conv(out)
        out = self.crossformer(out)
        feas = []
        for i in range(len(feats)):
            feas.append(torch.cat((F.interpolate(out, size=feats[i].shape[2:], mode='nearest'), feats[i]), dim=1))
        return tuple(feas)

    def forward(self, feats):
        feats = self.context_module(feats)
        cls_res, reg_res, hor_predict, ver_predict = multi_apply(self.forward_single, feats)
        level_num = len(cls_res)
        preds = [[cls_res[i], reg_res[i], hor_predict[i], ver_predict[i]] for i in range(level_num)]
        return preds

    def forward_single(self, x):
        x = x[:, 96:, :, :]
        rcab_predict = self.rcab(x)
        hor_predict, ver_predict, roam = self.loa(x)
        context = self.context_conv(x[:, :96, :, :])
        x = roam + rcab_predict + context
        cls_predict = self.out_conv_cls(x)
        reg_predict = self.out_conv_reg(x)
        return cls_predict, reg_predict, hor_predict, ver_predict

    def get_boundary(self, score_maps, img_metas, rescale):
        assert len(score_maps) == len(self.scales)

        boundaries = []
        for idx, score_map in enumerate(score_maps):
            scale = self.scales[idx]
            boundaries = boundaries + self._get_boundary_single(
                score_map, scale)

        # nms
        boundaries = poly_nms(boundaries, self.nms_thr)

        if rescale:
            boundaries = self.resize_boundary(
                boundaries, 1.0 / img_metas[0]['scale_factor'])

        results = dict(boundary_result=boundaries)
        return results

    def _get_boundary_single(self, score_map, scale):
        assert len(score_map) == 4
        assert score_map[1].shape[1] == 4 * self.fourier_degree + 2

        return decode(
            decoding_type=self.decoding_type,
            preds=score_map,
            fourier_degree=self.fourier_degree,
            num_reconstr_points=self.num_reconstr_points,
            scale=scale,
            alpha=self.alpha,
            beta=self.beta,
            text_repr_type=self.text_repr_type,
            score_thr=self.score_thr,
            nms_thr=self.nms_thr)
