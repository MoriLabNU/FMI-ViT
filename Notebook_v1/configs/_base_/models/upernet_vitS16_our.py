# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=(224, 224),
#     # mean=[123.675, 116.28, 103.53],
#     # std=[58.395, 57.12, 57.375],
    mean=[0, 0, 0],
    std=[255, 255, 255],
#     bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    #pretrained='/homes/yunhengwu/code/D1/CLSM_FM/dino-main/save_model_small16_wo_dino/student_mmseg_small16_wo_dino_checkpoint.pth',#'/homes/yunhengwu/code/D1/CLSM_FM/dino-main/DINO_weight/student_dino_deitsmall16_pretrain_full_checkpoint_mmseg.pth',#'/homes/yunhengwu/code/D1/CLSM_FM/dino-main/DINO_weight/student_dino_deitsmall16_pretrain_full_checkpoint.pth'
    backbone=dict(
        type='VisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint='/homes/yunhengwu/code/D1/CLSM_FM/dino-main/save_model_small16_wo_dino/teacher_mmseg_small16_wo_dino_checkpoint.pth'),
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
    #     frozen_exclude=[
    #     'layers.9.',
    #     'layers.10.',
    #     'layers.11.',
    #     'ln1.weight',
    #     'ln1.bias',
    # ],
        interpolate_mode='bicubic'),
    neck=dict(type='Feature2Pyramid', embed_dim=384, rescales=[4, 2, 1, 0.5]),
    # neck=dict(
    #     type='MultiLevelNeck',
    #     in_channels=[384, 384, 384, 384],
    #     out_channels=384,
    #     scales=[4, 2, 1, 0.5]),
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
    # decode_head=dict(
    #     type='FCNHead',
    #     in_channels=384,
    #     in_index=3,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=[
    #             dict(type='CrossEntropyLoss',use_sigmoid=False,loss_weight=1.0),
    #             #dict(type='FBContrastLoss')
    #         ]
    #         ),
    # model training and testing settings
    train_cfg=dict(),
    #test_cfg=dict(mode='whole'))  # yapf: disable
    test_cfg=dict(mode='slide', crop_size=(224,224), stride=(112, 112)))




# #loss_contrast=dict(
#     type='FBContrastLoss',
#     metric='cos',
#     alpha_min=2.5,
#     alpha_max=0.25,
#     loss_weight_min=1.0,
#     loss_weight_max=1.0,
#     reduction='mean',
#     temp=1.0)  # temp 是预留给 future soft scaling 的