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
        img_prefix='/data/cy/jsxs/val2017/',
        classes=classes,
        ann_file='/data/cy/jsxs/annotations/instances_val2017.json'),
    test=dict(
        img_prefix='/data/cy/jsxs/val2017/',
        classes=classes,
        ann_file='/data/cy/jsxs/annotations/instances_val2017.json'))

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1))
    )

load_from = 'checkpoints/cao_roi_faster.pth'
