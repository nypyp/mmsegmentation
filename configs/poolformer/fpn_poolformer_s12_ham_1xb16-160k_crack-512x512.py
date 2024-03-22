_base_ = [
    '../_base_/schedules/schedule_160k.py',
    '../_base_/datasets/crack.py',
    '../_base_/default_runtime.py'
]
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s12_3rdparty_32xb128_in1k_20220414-f8d83051.pth'  # noqa
custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)


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
        type='mmpretrain.PoolFormer',
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
        type='LightHamHead',
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=512,
        ham_channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=1,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

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
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]
