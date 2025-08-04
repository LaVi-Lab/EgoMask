##########################################
# Please setup the following paths before running the script
##########################################
gd_ckpt=${GROUNDING_DINO_CHECKPOINT}
gd_config=${GROUNDING_DINO_CONFIG}
sam2_ckpt=${SAM2p1_CHECKPOINT}
sam2_config=${SAM2p1_CONFIG}
##########################################

# scripts
INFER_SCRIPT=evaluation/groundedsam2/inference_egomask.py
EVAL_SCRIPT=evaluation/eval_egomask.py


# arguments
dataset_type=$1
SUBSET_NUM=8
vis_save_path=results/EgoMask-SAM2p1


common_args="--sam2_ckpt ${sam2_ckpt} \
    --sam2_config ${sam2_config} \
    --gd_ckpt ${gd_ckpt} \
    --gd_config ${gd_config} \
    --subset_num ${SUBSET_NUM} \
    --dataset_type ${dataset_type} 
    --vis_save_path ${vis_save_path} \
    --grounding_with_max_confidence"



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