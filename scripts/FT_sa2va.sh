# Training Script
bash tools/dist.sh train projects/llava_sam2/configs/sa2va_4b_FT_with_egomask-train.py 8

# Convert trained model to huggingface format
# python projects/llava_sam2/hf/convert_to_hf.py projects/llava_sam2/configs/sa2va_4b_FT_with_egomask-train.py --pth-model [PATH_TO_PTH_MODEL] --save-path [PATH_TO_SAVE_FOLDER]