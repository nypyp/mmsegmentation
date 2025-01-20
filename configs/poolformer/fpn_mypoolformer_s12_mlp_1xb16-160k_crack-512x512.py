_base_ = [
    '../_base_/schedules/schedule_160k.py',
    '../_base_/datasets/crack.py',
    '../_base_/default_runtime.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s12_3rdparty_32xb128_in1k_20220414-f8d83051.pth'  # noqa
# custom_imports = dict(
#     imports=['mmpretrain.models'], allow_failed_imports=False)


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
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='myPoolFormer',
        arch='s12',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file, prefix='backbone.'),
        in_patch_size=7,
        in_stride=4,
        in_pad=2,
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_rate=0.,
        drop_path_rate=0.,
        out_indices=(0, 2, 4, 6),
        frozen_stages=0,
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_dataloader = dict(batch_size=16)
train_cfg = dict(val_interval=16000)
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mDice','mFscore'])
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mDice','mFscore'])

# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
#     paramwise_cfg=dict(
#         custom_keys={
#             'pos_block': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.),
#             'head': dict(lr_mult=10.)
#         }))

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(
#         type='PolyLR',
#         power=1.0,
#         begin=1500,
#         end=160000,
#         eta_min=0.0,
#         by_epoch=False,
#     )
# ]
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001))
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=0.9,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]