_base_ = "base_org.py"
classes = ("box",)
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
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
    sampler=dict(train=dict(_delete_=True, type="GroupSampler")),
)
evaluation = dict(interval=25000)
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[500, 750])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=1000)
checkpoint_config = dict(by_epoch=False, interval=250, max_keep_ckpts=20)
fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
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
