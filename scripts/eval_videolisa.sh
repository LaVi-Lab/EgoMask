# scripts
INFER_SCRIPT=evaluation/videolisa/inference_egomask.py
EVAL_SCRIPT=evaluation/eval_egomask.py

# arguments
dataset_type=$1
model_path=$2

# export VIDEO_LISA_ROOT="VideoLISA" [PATH_TO_VIDEOLISA]

exp_name=$(basename ${model_path})

SUBSET_NUM=8
NUM_FRMS_SPARSE=32
NUM_FRMS_DENSE=4
VID_TOWER=openai/clip-vit-large-patch14-336
vis_save_path=results/EgoMask-VideoLISA-${exp_name}

common_args="--version ${model_path} \
    --vision_tower ${VID_TOWER} \
    --num_frames_dense ${NUM_FRMS_DENSE} \
    --num_frames_sparse ${NUM_FRMS_SPARSE} \
    --subset_num ${SUBSET_NUM} \
    --dataset_type ${dataset_type} \
    --vis_save_path ${vis_save_path}"



CUDA_VISIBLE_DEVICES=0 python ${INFER_SCRIPT} ${common_args} --subset_idx 0 &
CUDA_VISIBLE_DEVICES=1 python ${INFER_SCRIPT} ${common_args} --subset_idx 1 &
CUDA_VISIBLE_DEVICES=2 python ${INFER_SCRIPT} ${common_args} --subset_idx 2 &
CUDA_VISIBLE_DEVICES=3 python ${INFER_SCRIPT} ${common_args} --subset_idx 3 &
CUDA_VISIBLE_DEVICES=4 python ${INFER_SCRIPT} ${common_args} --subset_idx 4 &
CUDA_VISIBLE_DEVICES=5 python ${INFER_SCRIPT} ${common_args} --subset_idx 5 &
CUDA_VISIBLE_DEVICES=6 python ${INFER_SCRIPT} ${common_args} --subset_idx 6 &
CUDA_VISIBLE_DEVICES=7 python ${INFER_SCRIPT} ${common_args} --subset_idx 7 
wait
python ${EVAL_SCRIPT} --dataset_type ${dataset_type} --pred_path ${vis_save_path}