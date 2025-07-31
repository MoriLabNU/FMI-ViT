
exp_name = 'exp_vitS16'
save_dir = f'work_dirs3/{exp_name}'

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='TensorboardVisBackend'),dict(type='LocalVisBackend'),
            #     dict(type='WandbVisBackend',
            #  init_kwargs=dict(
            #      project='mmseg-fine-tune',
            #      name=exp_name,
            #  ))
             ]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer',save_dir=save_dir)


log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
