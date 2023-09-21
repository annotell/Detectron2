from functools import partial
import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.modeling import SimpleFeaturePyramid, DinoVisionTransformer
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .rcnn_fpn import model

# Base
embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
# Creates Simple Feature Pyramid from ViT backbone
model.backbone = L(SimpleFeaturePyramid)(
    net=L(DinoVisionTransformer)(  # Single-scale Dino ViT backbone
        img_size=518,
        patch_size=14,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(2.0, 1.0),
    norm="LN",
    square_pad=518,
)

# 2conv in RPN:
model.proposal_generator.head.conv_dims = [-1, -1]
