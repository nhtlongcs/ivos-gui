from stcn.backbone.mod_resnet import ResNetBackbone
from stcn.backbone.vision_transformer import SwinTransformer

def create_model(name, pretrained:bool = True, **kwargs):

    if 'resnet' in name:
        return ResNetBackbone(name, pretrained, **kwargs)

    if 'swin' in name:
        return SwinTransformer(
                  embed_dim=128,
                  depths=[2, 2, 18, 2],
                  num_heads=[4, 8, 16, 32],
                  window_size=7,
                  drop_path_rate=0.3,
                  out_indices=(0, 1, 2),
                  ape=False,
                  patch_norm=True,
                  frozen_stages=0,
                  use_checkpoint=False,
                  **kwargs)