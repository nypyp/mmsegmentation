_base_ = [
    '../_base_/models/fpn_poolformer_s12.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_320k.py',
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
    neck=dict(in_channels=[64, 128, 320, 512]),
    decode_head=dict(
        num_classes=2,
        out_channels=1,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
        )))

train_dataloader = dict(batch_size=16)
train_cfg = dict(val_interval=16000)
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
        end=320000,
        eta_min=0.0,
        by_epoch=False,
    )
]
