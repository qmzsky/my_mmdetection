_base_ = '../cao/cao_roi_faster.py'

# 修改数据集相关设置
dataset_type = 'CocoDataset'
classes = ('jsxs',)
data = dict(
    train=dict(
        img_prefix='/data/cy/jsxs/train2017/',
        classes=classes,
        ann_file='/data/cy/jsxs/annotations/instances_train2017.json'),
    val=dict(
        img_prefix='/data/cy/jsxs/train2017/',
        classes=classes,
        ann_file='/data/cy/jsxs/annotations/instances_train2017.json'),
    test=dict(
        img_prefix='/data/cy/jsxs/train2017/',
        classes=classes,
        ann_file='/data/cy/jsxs/annotations/instances_train2017.json'))

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1))
    )

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[80, 99])
runner = dict(type='EpochBasedRunner', max_epochs=100)

load_from = 'checkpoints/cao_roi_faster.pth'