_base_ = "base.py"
fold = 1
percent = 10
classes = ('box',)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        classes=classes,
        ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
        img_prefix="data/coco/train2017/",
    ),
)
model = dict(
    rpn_head=dict(
        anchor_generator=dict(
            scales=[
                3.356991467896006,
                4.532190435442829,
                5.703727111923104,
                7.860589377289129,
                10.558823804356944,
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
