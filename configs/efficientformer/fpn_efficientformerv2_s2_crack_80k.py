_base_ = [
    '../_base_/models/fpn_poolformer_s12.py',
    '../_base_/schedules/schedule_160k.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/crack.py'

]

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512),
    test_cfg=dict(size_divisor=32))

# model settings
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='efficientformerv2_s2_feat',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/nypyp/code/mmsegmentation/weights/eformer_s2_450.pth'
        ),
        drop_path_rate =0.02,
    ),
    neck=dict(in_channels=[32, 64, 144, 288]),
    decode_head=dict(num_classes=2))

train_dataloader = dict(batch_size=8)
#optimizer
optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001)
param_scheduler = [
    dict(
        type='PolyLR',
        power=0.9,
        begin=0,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]