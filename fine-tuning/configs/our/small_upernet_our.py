_base_ = [
    '../_base_/models/upernet_vitS16_our.py', 
    '../_base_/datasets/clsm_512.py', 
    '../_base_/default_runtime_vitS16.py', 
    '../_base_/schedules/schedule_60k_w_weight.py' 
]


fp16 = dict(loss_scale='dynamic') 

