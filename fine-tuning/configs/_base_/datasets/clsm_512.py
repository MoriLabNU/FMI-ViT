# dataset settings
dataset_type = 'CLSMDataset'
data_root = '/vessel_1'

img_scale = (512, 512)
crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize',scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size),  
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
    #dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

query_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize',scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size),  
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

query_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root= '/vessel_2',
        data_prefix=dict(
    img_path='images/training', seg_map_path='annotations/training'),
        pipeline=query_pipeline ))

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))



test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['all'])
test_evaluator = val_evaluator

custom_hooks = [
    dict(type='DualInputHook') #,
]

