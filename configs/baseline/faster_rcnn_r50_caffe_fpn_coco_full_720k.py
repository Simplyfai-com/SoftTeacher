_base_ = "base.py"

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        ann_file="data/coco/annotations/instances_train2017.json",
        img_prefix="data/coco/train2017/",
    ),
    sampler=dict(train=dict(type="GroupSampler")),
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
    )
)
optimizer = dict(lr=0.02)
lr_config = dict(step=[250, 500, 750])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=1000)
