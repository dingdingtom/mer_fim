export PYTHONPATH=${pwd}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
num_device=4

dataset_dir=Your/Dataset/Dir/

python trainer.py \
    --bs_train 200 \
    --dataset_dir ${dataset_dir} \
    --lr 0.0002 \
    --name_dataset CHERMA \
    --name_save checkpoint_cherma \
    --num_device ${num_device} \
    --num_epoch_max 16
