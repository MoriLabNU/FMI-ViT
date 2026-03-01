# Copyright (c) OpenMMLab. All rights reserved.
# yapf: disable
from .ade import ADE20KDataset
from .nyu import NYUDataset
from .clsm import CLSMDataset
from .drive import DRIVEDataset


# yapf: enable
__all__ = [
    'ADE20KDataset', 'NYUDataset', 'CLSMDataset', 'DRIVEDataset', 
]
