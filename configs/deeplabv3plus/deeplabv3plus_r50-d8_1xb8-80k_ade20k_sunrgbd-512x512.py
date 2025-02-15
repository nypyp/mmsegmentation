_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/ade20k_sunrgbd.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=37),
    auxiliary_head=dict(num_classes=37))
    # backbone=dict(
    #     init_cfg=dict(
    #         type='Pretrained',
    #         checkpoint='open-mmlab://resnet50_v1c',
    #         prefix='backbone.'
    #     )
    # )
        

train_dataloader = dict(
    batch_size=8,
)

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend'),
                dict(type='WandbVisBackend')]
)

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook',draw=True,interval=50)
)