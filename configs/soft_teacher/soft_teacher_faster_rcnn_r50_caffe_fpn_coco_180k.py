_base_ = "base.py"
fold = 1
percent = 10
classes = ("box",)
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
    sampler=dict(train=dict(_delete_=True, type="GroupSampler")),
)

model = dict(
    rpn_head=dict(
        anchor_generator=dict(
            scales=[
                13.770410151757826,
                15.964492326546731,
                18.727237019627474,
                19.836958197941414,
                22.108723664209517,
            ],
            ratios=[
                0.692972000449792,
                0.9578803670273774,
                1.0816754348833928,
                1.3359649417858739,
                1.5288758535865943,
            ],
            strides=[4, 8, 16, 32, 64],
        )
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        )
    ),
)

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=1000)
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
