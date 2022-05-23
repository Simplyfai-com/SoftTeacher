for FOLD in 1;
do
  bash tools/dist_train_partially.sh semi ${FOLD} 1 8
done