# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine.fileio as fileio
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample

import torch

@HOOKS.register_module()
class SegVisualizationHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        if self.draw is False or mode == 'train':
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                img_path = output.img_path
                img_bytes = fileio.get(
                    img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                window_name = f'{mode}_{osp.basename(img_path)}'

 
                self._visualizer.add_datasample(
                    window_name,
                    img,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter)



@HOOKS.register_module()
class DualInputHook(Hook):
    def before_train(self, runner):
        self.query_loader = runner.build_dataloader(runner.cfg.query_dataloader)
        self.query_iter = iter(self.query_loader)

    def before_train_iter(self, runner, batch_idx, data_batch):
        # 获取 query 图像 batch
        try:
            query_batch = next(self.query_iter)
        except StopIteration:
            self.query_iter = iter(self.query_loader)
            query_batch = next(self.query_iter)

        # 注入 query 图像和 meta 信息
        data_batch['query_img'] = query_batch['inputs']
        data_batch['query_meta'] = query_batch.get('data_samples', None)

        # 提取 query label (annotation)，注意结构为 SegDataSample
        if 'data_samples' in query_batch:
            query_labels = torch.stack([
                sample.gt_sem_seg.data.squeeze(0)  # squeeze channel if needed
                for sample in query_batch['data_samples']
            ])
            data_batch['query_label'] = query_labels  # [B, H, W]

# @HOOKS.register_module()
# class DualInputHook(Hook):
#     def before_train(self, runner):
#         self.query_loader = runner.build_dataloader(runner.cfg.query_dataloader)
#         self.query_iter = iter(self.query_loader)

#     def before_train_iter(self, runner, batch_idx, data_batch):
#         # 获取 query 图像 batch
#         try:
#             query_batch = next(self.query_iter)
#         except StopIteration:
#             self.query_iter = iter(self.query_loader)
#             query_batch = next(self.query_iter)

#         # 将 query 图像注入主 batch
#         data_batch['query_img'] = query_batch['inputs']
#         data_batch['query_meta'] = query_batch.get('data_samples', None)

# @HOOKS.register_module()
# class EnableLoss2Hook(Hook):
#     def __init__(self, enable_after_iter=1000):
#         self.enable_after_iter = enable_after_iter

#     def before_train_iter(self, runner, batch_idx, data_batch):
#         model = runner.model
#         if runner.iter >= self.enable_after_iter:
#             if hasattr(model, 'module'):
#                 model.module.decode_head.enable_loss2 = True
#             else:
#                 model.decode_head.enable_loss2 = True