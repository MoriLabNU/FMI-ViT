from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class CLSMDataset(BaseSegDataset):
    """Binary segmentation dataset for CLSM (e.g., vessel vs. background).

    Assumes:
        - Class 0: background
        - Class 1: foreground (e.g., vessel)
        - Uses '.png' as segmentation map suffix.
        - Uses '.jpg' as image suffix.
    """
    METAINFO = dict(
        classes=('background', 'foreground'),
        palette=[[120, 120, 120], [6, 230, 230]]  # black for background, white for vessel
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
