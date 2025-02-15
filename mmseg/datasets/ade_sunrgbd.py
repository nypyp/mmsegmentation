# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ADE20K_sunrgbd_Dataset(BaseSegDataset):
    """sunrgbd dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 37 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=(
            'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 
            'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds',
            'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror',
            'floor_mat', 'clothes', 'ceiling', 'books', 'fridge', 'tv',
            'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
            'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub',
            'bag'
        ),
        palette=[[0, 1, 0], [131, 128, 1], [198, 120, 12], [67, 0, 143],
                 [191, 0, 126], [0, 200, 0], [121, 193, 0], [0, 55, 122],
                 [201, 60, 128], [195, 187, 136], [132, 0, 193], [72, 3, 48],
                 [70, 0, 195], [199, 127, 190], [190, 194, 58], [62, 66, 174],
                 [187, 191, 192], [30, 2, 1],  [175, 0, 0], [101, 0, 141],
                 [87, 126, 125], [218, 124, 124], [24, 61, 0], [255, 188, 0],
                 [84, 68, 131], [218, 59, 123], [240, 191, 132], [142, 5, 61],
                 [19, 133, 57], [120, 0, 0], [120, 120, 120], [180, 120, 120],
                 [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80],
                 [140, 140, 140]
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
