import torch
import torch.nn as nn

from ..builder import HEADS
from .standard_roi_head import StandardRoIHead


class RoiAttention(nn.Module):

    def __init__(self):
        super(RoiAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(256, 4)
        self.bn = nn.BatchNorm1d(49)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = x.flatten(-2)
        x = x.permute(0, 2, 1)
        x = x.permute(1, 0, 2)

        residual = x
        x = self.self_attn(x, x, x)[0]
        x += residual

        x = x.permute(1, 0, 2)
        x = self.bn(x)
        x = self.gelu(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, 256, 7, 7)
        return x

@HEADS.register_module()
class CaoRoiHeadV3(StandardRoIHead):

    def __init__(self, **kwargs):
        super(CaoRoiHeadV3, self).__init__(**kwargs)
        self.roiAttention = RoiAttention()

    def _bbox_forward(self, x, rois):
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        bbox_feats = self.roiAttention(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        return bbox_results