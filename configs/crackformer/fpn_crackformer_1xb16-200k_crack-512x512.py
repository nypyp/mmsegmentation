_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_200k.py',
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
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='crackformer',
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='/xxx',
        #     prefix='backbone.',
        # ),
    ),
    decode_head=dict(
        type='FakeHead',
        in_channels=5,
        out_channels=1,
        channels = 5,
        dropout_ratio=0.1,
        align_corners=False,
        num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_dataloader = dict(batch_size=4)
train_cfg = dict(val_interval=20000)
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mDice','mFscore'])
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mDice','mFscore'])
# optimizer
# optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
# optimizer_config = dict()
# # learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001))
param_scheduler = [
    dict(
        type='PolyLR',
        power=0.9,
        begin=0,
        end=200000,
        eta_min=0.0,
        by_epoch=False,
    )
]
