_base_ = "base.py"
fold = 1
percent = 50
classes = ("cancer",)
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=1,
    train=dict(
        sup=dict(
            type="CocoDataset",
            classes=classes,
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="data/coco/train2017/",
        ),
        unsup=dict(
            type="CocoDataset",
            classes=classes,
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="data/coco/train2017/",
        ),
    ),
    val=dict(
        type="CocoDataset",
        classes=classes,
        ann_file="data/coco/annotations/instances_val2017.json",
        img_prefix="data/coco/val2017/",
    ),
    test=dict(
        type="CocoDataset",
        classes=classes,
        ann_file="data/coco/annotations/instances_test2017.json",
        img_prefix="data/coco/test2017/",
    ),
    sampler=dict(train=dict(_delete_=True, type="GroupSampler")),
)

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        )
    ),
)

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=1000)
lr_config = dict(step=[250, 500, 750])
checkpoint_config = dict(by_epoch=False, interval=250, max_keep_ckpts=20)
work_dir = "work_dirs/cell_new/30_30/${cfg_name}/${percent}"
log_config = dict(
    interval=100,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="pre_release",
                name="${cfg_name}",
                config=dict(
                    fold="${fold}",
                    percent="${percent}",
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)
