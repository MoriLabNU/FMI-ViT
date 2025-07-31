# optimizer
optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05)

optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None,
        paramwise_cfg = dict(
                            custom_keys={'backbone': dict(lr_mult=0.1)}))
# # learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=8000,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=8000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, max_keep_ckpts=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook',draw=True,interval=1))

