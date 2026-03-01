_base_ = [
    '../_base_/models/upernet_vitS16_our.py',  # 使用 MAE 作为主干网络的 UPerNet 模型配置
    '../_base_/datasets/clsm_512_vessel.py', # CLSM 数据集配置，图像尺寸为 518x518
    '../_base_/default_runtime_vitS16.py',  # 默认运行设置，例如日志记录、检查点保存
    '../_base_/schedules/schedule_60k_test.py' # 训练计划，设定训练迭代次数为 60,000
]



# mixed precision 
fp16 = dict(loss_scale='dynamic')  # 自动动态缩放 loss 值防止溢出

# # By default, models are trained on 8 GPUs with 2 images per GPU
# train_dataloader = dict(batch_size=128)  # 训练时每个 GPU 处理 8 张图片
# val_dataloader = dict(batch_size=32)  # 验证时每个 GPU 处理 1 张图片
# test_dataloader = val_dataloader  # 测MSC试使用验证的配置

