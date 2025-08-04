# Reproduce the final result (Tab. 1, Tab. 2, and Tab. 3)
# The model is trained using 8 nodes (64 A10 GPUs), each A10 GPU has 24GB memory
master_port=29517

version="MBZUAI/LLaVA-Phi-3-mini-4k-instruct"
weight="ZechenBai/VideoLISA-3.8B"

dataset_dir="videolisa_data"
vision_pretrained="sam_vit_h_4b8939.pth"

vision_tower="openai/clip-vit-large-patch14-336"
exp_name="video-lisa-3.8b-FT-with-egomask-train"

CUDA_LAUNCH_BLOCKING=1 deepspeed --include localhost:0,1,2,3 --master_port ${master_port} train_joint.py \
  --version=${version}  \
  --dataset_dir=${dataset_dir} \
  --vision_pretrained=${vision_pretrained} \
  --vision-tower=${vision_tower} \
  --weight=${weight} \
  --exp_name=${exp_name} \
  --num_frames_sparse=32 \
  --num_frames_dense=4 \
  --num_classes_per_sample=1 \
  --epochs=20 \
  --steps_per_epoch=500 \
  --batch_size=4 \
  --lr=0.00003 \
  --grad_accumulation_steps=1 \
  --model_max_length=2048 \
  --dataset="sem_seg,refer_seg,reason_seg,vos,ref_vos,davis,egomask" \
  --sem_seg_data="ade20k,cocostuff,pascal_part,paco_lvis" \
  --refer_seg_data="refclef,refcoco,refcoco+,refcocog" \
  --vos_data="ytvos" \
  --ref_vos_data="refer_youtube_vos,mevis" \
  --sample_rates="2,1,1,2,2,1,36" 2>&1 | tee FT_with_egomask-train.log

cd runs/${exp_name}/ckpt_model
python zero_to_fp32.py . ../pytorch_model.bin

cd ../../../
CUDA_VISIBLE_DEVICES=${local_rank} python merge_lora_weights_and_save_hf_model.py \
  --version=${weight} \
  --vision-tower=${vision_tower} \
  --vision_pretrained=${vision_pretrained} \
  --weight="runs/${exp_name}/pytorch_model.bin" \
  --save_path="runs/${exp_name}/merged"