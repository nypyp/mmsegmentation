_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/ade20k_indoor.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://contrib/mobilenet_v3_large',
    backbone=dict(
        _delete_=True,
        type='MobileNetV3',
        arch='large',
        out_indices=(0, 3, 16),
        norm_cfg=norm_cfg),
    decode_head=dict(
        _delete_=True,
        type='AIFIHead',
        in_channels=960,
        channels=480,
        c1_in_channels=24,
        c1_channels=48,
        in_index=2,
        dropout_ratio=0.1,
        num_classes=58,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(
        _delete_=True,
        type='FCNHead',
        in_channels=24,
        in_index=1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=58,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
)
# dataset settings
train_dataloader = dict(
    batch_size=16,
    )

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend'),
                dict(type='WandbVisBackend')]
)




