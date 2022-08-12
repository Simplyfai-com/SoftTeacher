_base_ = "base.py"
classes = ("cancer",)

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=1,
    train=dict(
        type="CocoDataset",
        classes=classes,
        ann_file="data/coco_50/annotations/instances_train2017.json",
        img_prefix="data/coco_50/train2017/",
    ),
    test=dict(
        type="CocoDataset",
        classes=classes,
        ann_file="data/coco_50/annotations/instances_test2017.json",
        img_prefix="data/coco_50/test2017/",
    ),
    sampler=dict(train=dict(type="GroupSampler")),
)
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        ),
    ),
)

optimizer = dict(lr=0.002)
lr_config = dict(step=[1000, 2000, 4000, 8000])
evaluation = dict(interval=1000)
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=10000)
work_dir = "work_dirs/cell_new/50_50/${cfg_name}"
