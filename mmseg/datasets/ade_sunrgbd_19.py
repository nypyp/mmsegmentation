# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ADE20K_sunrgbd_19_Dataset(BaseSegDataset):
    """sunrgbd dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 37 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=(
            'wall', 'floor', 'bed', 'chair',
            'table', 'door', 'window', 'bookshelf',
            'picture', 'curtain', 'ceiling',
            'fridge', 'tv', 'box', 'person',
            'toilet', 'sink', 'lamp', 'bathtub'
        ),
        palette=[[218, 59, 123], [75, 7, 46], [0, 200, 0], [67, 0, 143],
                 [191, 0, 126], [198, 120, 12], [121, 193, 0], [0, 55, 122],
                 [201, 60, 128], [195, 187, 136], [72, 3, 48],
                 [70, 0, 195], [199, 127, 190], [190, 194, 58], [62, 66, 174],
                 [187, 191, 192], [30, 2, 1],  [175, 0, 0], [101, 0, 141],
        ]
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
