# Copyright (c) OpenMMLab. All rights reserved.
import collections
import copy
from typing import List, Optional, Sequence, Union

from mmengine.dataset import ConcatDataset, force_full_init

from mmseg.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class MultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup.

    Args:
        dataset (ConcatDataset or dict): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
    """

    def __init__(self,
                 dataset: Union[ConcatDataset, dict],
                 pipeline: Sequence[dict],
                 skip_type_keys: Optional[List[str]] = None,
                 lazy_init: bool = False) -> None:
        assert isinstance(pipeline, collections.abc.Sequence)

        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, ConcatDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`ConcatDataset` instance, but got {type(dataset)}')

        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = TRANSFORMS.build(transform)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self._metainfo = self.dataset.metainfo
        self.num_samples = len(self.dataset)

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset.

        Returns:
            dict: The meta information of multi-image-mixed dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types):
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indices'):
                indices = transform.get_indices(self.dataset)
                if not isinstance(indices, collections.abc.Sequence):
                    indices = [indices]
                mix_results = [
                    copy.deepcopy(self.dataset[index]) for index in indices
                ]
                results['mix_results'] = mix_results

            results = transform(results)

            if 'mix_results' in results:
                results.pop('mix_results')

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys.

        It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys




# @DATASETS.register_module()
# class RepeatDataset:
#     """A wrapper of repeated dataset.

#     The length of repeated dataset will be `times` larger than the original
#     dataset. This is useful when the data loading time is long but the dataset
#     is small. Using RepeatDataset can reduce the data loading time between
#     epochs.

#     Note:
#         ``RepeatDataset`` should not inherit from ``BaseDataset`` since
#         ``get_subset`` and ``get_subset_`` could produce ambiguous meaning
#         sub-dataset which conflicts with original dataset. If you want to use
#         a sub-dataset of ``RepeatDataset``, you should set ``indices``
#         arguments for wrapped dataset which inherit from ``BaseDataset``.

#     Args:
#         dataset (BaseDataset or dict): The dataset to be repeated.
#         times (int): Repeat times.
#         lazy_init (bool): Whether to load annotation during
#             instantiation. Defaults to False.
#     """

#     def __init__(self,
#                  dataset: Union[BaseDataset, dict],
#                  times: int,
#                  lazy_init: bool = False):
#         self.dataset: BaseDataset
#         if isinstance(dataset, dict):
#             self.dataset = DATASETS.build(dataset)
#         elif isinstance(dataset, BaseDataset):
#             self.dataset = dataset
#         else:
#             raise TypeError(
#                 'elements in datasets sequence should be config or '
#                 f'`BaseDataset` instance, but got {type(dataset)}')
#         self.times = times
#         self._metainfo = self.dataset.metainfo

#         self._fully_initialized = False
#         if not lazy_init:
#             self.full_init()

#     @property
#     def metainfo(self) -> dict:
#         """Get the meta information of the repeated dataset.

#         Returns:
#             dict: The meta information of repeated dataset.
#         """
#         return copy.deepcopy(self._metainfo)

#     def full_init(self):
#         """Loop to ``full_init`` each dataset."""
#         if self._fully_initialized:
#             return

#         self.dataset.full_init()
#         self._ori_len = len(self.dataset)
#         self._fully_initialized = True

#     @force_full_init
#     def _get_ori_dataset_idx(self, idx: int) -> int:
#         """Convert global index to local index.

#         Args:
#             idx: Global index of ``RepeatDataset``.

#         Returns:
#             idx (int): Local index of data.
#         """
#         return idx % self._ori_len

#     @force_full_init
#     def get_data_info(self, idx: int) -> dict:
#         """Get annotation by index.

#         Args:
#             idx (int): Global index of ``ConcatDataset``.

#         Returns:
#             dict: The idx-th annotation of the datasets.
#         """
#         sample_idx = self._get_ori_dataset_idx(idx)
#         return self.dataset.get_data_info(sample_idx)

#     def __getitem__(self, idx):
#         if not self._fully_initialized:
#             print_log(
#                 'Please call `full_init` method manually to accelerate the '
#                 'speed.',
#                 logger='current',
#                 level=logging.WARNING)
#             self.full_init()

#         sample_idx = self._get_ori_dataset_idx(idx)
#         return self.dataset[sample_idx]

#     @force_full_init
#     def __len__(self):
#         return self.times * self._ori_len

#     def get_subset_(self, indices: Union[List[int], int]) -> None:
#         """Not supported in ``RepeatDataset`` for the ambiguous meaning of sub-
#         dataset."""
#         raise NotImplementedError(
#             '`RepeatDataset` dose not support `get_subset` and '
#             '`get_subset_` interfaces because this will lead to ambiguous '
#             'implementation of some methods. If you want to use `get_subset` '
#             'or `get_subset_` interfaces, please use them in the wrapped '
#             'dataset first and then use `RepeatDataset`.')

#     def get_subset(self, indices: Union[List[int], int]) -> 'BaseDataset':
#         """Not supported in ``RepeatDataset`` for the ambiguous meaning of sub-
#         dataset."""
#         raise NotImplementedError(
#             '`RepeatDataset` dose not support `get_subset` and '
#             '`get_subset_` interfaces because this will lead to ambiguous '
#             'implementation of some methods. If you want to use `get_subset` '
#             'or `get_subset_` interfaces, please use them in the wrapped '
#             'dataset first and then use `RepeatDataset`.')

