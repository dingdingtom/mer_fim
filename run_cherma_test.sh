export PYTHONPATH=${pwd}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

dataset_dir=Your/Dataset/Dir/
path_ckpt=Your/Path/.ckpt

python trainer.py \
    --dataset_dir ${dataset_dir} \
    --mode test \
    --name_dataset CHERMA \
    --need_resume 1 \
    --path_ckpt ${path_ckpt}
