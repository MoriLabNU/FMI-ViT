
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=(224, 224),
    mean=[0, 0, 0],
    std=[255, 255, 255],
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint='checkpoint.pth'),
        img_size=(224, 224),
        patch_size=16,
        in_channels=3,
        embed_dims=384,
        num_layers=12,
        num_heads=6,
        mlp_ratio=4,
        out_indices=(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        frozen_exclude=[],
        interpolate_mode='bicubic'),
    neck=dict(type='Feature2Pyramid', embed_dim=384, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
                dict(type='CrossEntropyLoss',use_sigmoid=False,loss_weight=1.0),
                dict(type='FBContrastLoss')
            ]),
    train_cfg=dict(),

    test_cfg=dict(mode='slide', crop_size=(224,224), stride=(112, 112)))

