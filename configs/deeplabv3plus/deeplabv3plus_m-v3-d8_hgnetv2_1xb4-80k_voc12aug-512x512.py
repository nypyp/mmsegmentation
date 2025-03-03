_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://contrib/mobilenet_v3_large',
    backbone=dict(
        _delete_=True,
        type='HGNetV2',  # 使用注册的HGNetV2
        arch='L',        # 选择架构版本（S/M/L/X/H）
        use_lab=True,    # 是否启用LearnableAffineBlock
        return_idx=[0, 1, 2, 3],  # 选择输出的阶段索引
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=516,
        dilations=(1, 12, 24, 36),
        c1_in_channels=24,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # decode_head=dict(
    #     _delete_=True,
    #     type='AIFIHead',
    #     in_channels=960,
    #     channels=256,
    #     c1_in_channels=24,
    #     c1_channels=48,
    #     transformer_channels=2048,
    #     in_index=2,
    #     dropout_ratio=0.1,
    #     num_classes=21,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    # ),
    auxiliary_head=dict(
        _delete_=True,
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # auxiliary_head=None,
)
# dataset settings
train_dataloader = dict(
    batch_size=4,
    )

# visualizer = dict(
#     vis_backends=[dict(type='LocalVisBackend'),
#                 dict(type='TensorboardVisBackend'),
#                 dict(type='WandbVisBackend')]
# )

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook',draw=True,interval=50)
)




