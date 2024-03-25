_base_ = [
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512,512),
    test_cfg=dict(size_divisor=32))
# model settings
model = dict(
    type='EncoderDecoder',
    data_preprocessor = data_preprocessor,
    backbone=dict(
        type='efficientformerv2_s1_feat',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/nypyp/code/mmsegmentation/weights/eformer_s1_450.pth',
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[32, 48, 120, 224],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_dataloader = dict(batch_size=8)
train_cfg = dict(val_interval=10000)
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mDice','mFscore'])
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mDice','mFscore'])
# optimizer
# optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001))
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