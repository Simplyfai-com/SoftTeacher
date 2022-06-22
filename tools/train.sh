JOB_TYPE=semi
FOLD=1
PERCENT=10
GPUS=1
# bash tools/dist_train_partially.sh semi ${JOB_TYPE} ${FOLD} ${PERCENT} ${GPUS}

# Let's run the semi-supervided train directly since this is a single GPU machine
python3 tools/train.py \
    configs/soft_teacher/soft_teacher_faster_rcnn_r_leiver.py \
    --work-dir work_dirs/new_dataset/ \
    --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5}