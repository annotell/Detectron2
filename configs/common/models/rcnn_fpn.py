from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone import BasicStem, FPN, ResNet
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
)

from ..data.constants import constants

model = L(GeneralizedRCNN)(
    backbone=L(FPN)(
        bottom_up=L(ResNet)(
            stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
            stages=L(ResNet.make_default_stages)(
                depth=50,
                stride_in_1x1=True,
                norm="FrozenBN",
            ),
            out_features=["res2", "res3", "res4", "res5"],
        ),
        in_features="${.bottom_up.out_features}",
        out_channels=256,
        top_block=L(LastLevelMaxPool)(),
    ),
    proposal_generator=L(RPN)(
        in_features=["p2", "p3"],
        head=L(StandardRPNHead)(in_channels=256, num_anchors=2),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32], [64]],
            aspect_ratios=[0.5, 1.0],
            strides=[
                7,
                14,
            ],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
    ),
    roi_heads=L(StandardROIHeads)(
        num_classes=80,
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=L(Matcher)(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        box_in_features=["p2", "p3"],
        box_pooler=L(ROIPooler)(
            output_size=7,
            scales=(1.0 / 7, 1.0 / 14),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN",
        ),
        box_predictor=L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes="${..num_classes}",
        ),
    ),
    pixel_mean=[255 * 0.485, 255 * 0.456, 255 * 0.406],
    pixel_std=[255 * 0.229, 255 * 0.224, 255 * 0.225],
    input_format="RBG",
)
