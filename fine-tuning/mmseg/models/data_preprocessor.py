# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS
from mmseg.utils import stack_batch


@MODELS.register_module()
class SegDataPreProcessor(BaseDataPreprocessor):
    """Image pre-processor for segmentation tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.


    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
        test_cfg (dict, optional): The padding size config in testing, if not
            specify, will use `size` and `size_divisor` params as default.
            Defaults to None, only supports keys `size` or `size_divisor`.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    # def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
    #     """Perform normalization、padding and bgr2rgb conversion based on
    #     ``BaseDataPreprocessor``.

    #     Args:
    #         data (dict): data sampled from dataloader.
    #         training (bool): Whether to enable training time augmentation.

    #     Returns:
    #         Dict: Data in the same format as the model input.
    #     """
    #     data = self.cast_data(data)  # type: ignore
    #     inputs = data['inputs']
    #     data_samples = data.get('data_samples', None)
    #     # TODO: whether normalize should be after stack_batch
    #     if self.channel_conversion and inputs[0].size(0) == 3:
    #         inputs = [_input[[2, 1, 0], ...] for _input in inputs]

    #     inputs = [_input.float() for _input in inputs]
    #     if self._enable_normalize:
    #         inputs = [(_input - self.mean) / self.std for _input in inputs]

    #     if training:
    #         assert data_samples is not None, ('During training, ',
    #                                           '`data_samples` must be define.')
    #         inputs, data_samples = stack_batch(
    #             inputs=inputs,
    #             data_samples=data_samples,
    #             size=self.size,
    #             size_divisor=self.size_divisor,
    #             pad_val=self.pad_val,
    #             seg_pad_val=self.seg_pad_val)

    #         if self.batch_augments is not None:
    #             inputs, data_samples = self.batch_augments(
    #                 inputs, data_samples)
    #     else:
    #         img_size = inputs[0].shape[1:]
    #         assert all(input_.shape[1:] == img_size for input_ in inputs),  \
    #             'The image size in a batch should be the same.'
    #         # pad images when testing
    #         if self.test_cfg:
    #             inputs, padded_samples = stack_batch(
    #                 inputs=inputs,
    #                 size=self.test_cfg.get('size', None),
    #                 size_divisor=self.test_cfg.get('size_divisor', None),
    #                 pad_val=self.pad_val,
    #                 seg_pad_val=self.seg_pad_val)
    #             for data_sample, pad_info in zip(data_samples, padded_samples):
    #                 data_sample.set_metainfo({**pad_info})
    #         else:
    #             inputs = torch.stack(inputs, dim=0)

    #     return dict(inputs=inputs, data_samples=data_samples)
    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization, padding and color conversion for inputs and query_img.

        Args:
            data (dict): A batch of data sampled from dataloader.
            training (bool): Whether it's in training mode.

        Returns:
            Dict: Model inputs, including query_img if provided.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)

        # Support image channel conversion
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        # To float32
        inputs = [_input.float() for _input in inputs]

        # Normalize if enabled
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        # Stack + pad inputs
        if training:
            assert data_samples is not None, ('During training, `data_samples` must be defined.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(inputs, data_samples)
        else:
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs), \
                'The image size in a batch should be the same.'
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)


        if training and 'query_img' in data and 'query_label' in data:
            query_imgs = data['query_img']
            query_labels = data['query_label']

            # --- 图像预处理 ---
            if isinstance(query_imgs, list):
                query_imgs = [img.float() for img in query_imgs]
                if self.channel_conversion and query_imgs[0].size(0) == 3:
                    query_imgs = [img[[2, 1, 0], ...] for img in query_imgs]
                if self._enable_normalize:
                    query_imgs = [(img - self.mean) / self.std for img in query_imgs]
                query_imgs = torch.stack(query_imgs, dim=0)

            elif isinstance(query_imgs, torch.Tensor):
                query_imgs = query_imgs.float()
                if self.channel_conversion and query_imgs.size(1) == 3:
                    query_imgs = query_imgs[:, [2, 1, 0], :, :]
                if self._enable_normalize:
                    query_imgs = (query_imgs - self.mean) / self.std

            # --- 标签处理 ---
            if isinstance(query_labels, torch.Tensor) and query_labels.dim() == 3:
                query_labels = query_labels.unsqueeze(1)  # [B, 1, H, W]

            # --- 一次性注入 query_img 和 query_label ---
            for i in range(len(data_samples)):
                data_samples[i].set_data({
                    'query_img': query_imgs[i],
                    'query_label': query_labels[i]
                })


            # # ✅ 可视化：仅保存前10张用于验证
            # if self.training and len(data_samples) > 0:
            #     import os
            #     import torchvision.transforms.functional as TF
            #     from PIL import Image
            #     import numpy as np

            #     def colorize_mask(mask_tensor):
            #         mask = mask_tensor.numpy()
            #         color_map = np.array([
            #             [0, 0, 0],
            #             [255, 0, 0],
            #             [0, 255, 0],
            #             [0, 0, 255],
            #         ], dtype=np.uint8)
            #         return Image.fromarray(color_map[mask])

            #     def overlay_mask_on_image(image_pil, mask_tensor, alpha=0.5):
            #         mask_pil = colorize_mask(mask_tensor)
            #         return Image.blend(image_pil.convert("RGBA"), mask_pil.convert("RGBA"), alpha)

            #     save_dir = './debug_query_pairs'
            #     os.makedirs(save_dir, exist_ok=True)

            #     for i in range(min(10, len(data_samples))):
            #         query_img = data_samples[i].get('query_img')  # [3, H, W]
            #         query_label = data_samples[i].get('query_label')  # [1, H, W] or [H, W]

            #         img_pil = TF.to_pil_image(query_img.cpu())
            #         label_tensor = query_label.squeeze(0).cpu().byte()
            #         label_gray_pil = TF.to_pil_image(label_tensor)
            #         label_color_pil = colorize_mask(label_tensor)

            #         img_pil.save(os.path.join(save_dir, f'query_img_{i}.png'))
            #         label_gray_pil.save(os.path.join(save_dir, f'query_label_{i}.png'))

            #         concat = Image.new('RGB', (img_pil.width * 2, img_pil.height))
            #         concat.paste(img_pil, (0, 0))
            #         concat.paste(label_color_pil, (img_pil.width, 0))
            #         concat.save(os.path.join(save_dir, f'pair_vis_{i}.png'))

            #         overlayed = overlay_mask_on_image(img_pil, label_tensor)
            #         overlayed.save(os.path.join(save_dir, f'overlay_{i}.png'))
            #     exit()


        return dict(
            inputs=inputs,
            data_samples=data_samples,
        )
 